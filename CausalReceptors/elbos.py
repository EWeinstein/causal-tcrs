import torch
import pyro
from pyro.infer import JitTrace_ELBO
from pyro.infer import SVI
import pyro.poutine as poutine

from pyro.distributions.util import is_identically_zero
import weakref
from pyro.infer.util import (
    MultiFrameTensor,
    get_plate_stacks)

def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_plate_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r

class CudaJitTrace_ELBO(JitTrace_ELBO):

    def loss_and_surrogate_loss(self, model, guide, *args, **kwargs):
        kwargs["_pyro_model_id"] = id(model)
        kwargs["_pyro_guide_id"] = id(guide)
        if getattr(self, "_loss_and_surrogate_loss", None) is None:

            # build a closure for loss_and_surrogate_loss
            weakself = weakref.ref(self)

            @pyro.ops.jit.trace(
                ignore_warnings=self.ignore_jit_warnings, jit_options=self.jit_options
            )
            def loss_and_surrogate_loss(*args, **kwargs):
                kwargs.pop("_pyro_model_id")
                kwargs.pop("_pyro_guide_id")
                self = weakself()
                loss = 0.0
                surrogate_loss = 0.0
                for model_trace, guide_trace in self._get_traces(
                        model, guide, args, kwargs
                ):
                    elbo_particle = 0
                    surrogate_elbo_particle = 0
                    log_r = None

                    # compute elbo and surrogate elbo
                    for name, site in model_trace.nodes.items():

                        if site["type"] == "sample":
                            elbo_particle = elbo_particle + site["log_prob_sum"]
                            surrogate_elbo_particle = (
                                    surrogate_elbo_particle + site["log_prob_sum"]
                            )

                    for name, site in guide_trace.nodes.items():
                        if site["type"] == "sample":
                            log_prob, score_function_term, entropy_term = site[
                                "score_parts"
                            ]
                            elbo_particle = elbo_particle - site["log_prob_sum"]

                            if not is_identically_zero(entropy_term):
                                surrogate_elbo_particle = (
                                        surrogate_elbo_particle - entropy_term.sum()
                                )

                            if not is_identically_zero(score_function_term):
                                if log_r is None:
                                    log_r = _compute_log_r(model_trace, guide_trace)
                                site = log_r.sum_to(site["cond_indep_stack"])
                                surrogate_elbo_particle = (
                                        surrogate_elbo_particle
                                        + (site * score_function_term).sum()
                                )

                    loss = loss - elbo_particle / self.num_particles
                    surrogate_loss = (
                            surrogate_loss - surrogate_elbo_particle / self.num_particles
                    )

                return loss, surrogate_loss

            self._loss_and_surrogate_loss = loss_and_surrogate_loss

        return self._loss_and_surrogate_loss(*args, **kwargs)

    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss, surrogate_loss = self.loss_and_surrogate_loss(
            model, guide, *args, **kwargs
        )
        surrogate_loss.backward()
        return loss


class CudaSVI(SVI):

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        return loss