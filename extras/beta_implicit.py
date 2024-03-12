from dataclasses import dataclass
import jax
from tensorflow_probability.substrates import jax as tfp
from adevjax import ADEVPrimitive
from genjax.vi import ADEVDistribution

tfd = tfp.distributions


# Defining a new primitive.
@dataclass
class BetaIMPLICIT(ADEVPrimitive):
    # `flatten` is a method which is required to register this type as
    # a JAX PyTree type.
    def flatten(self):
        return (), ()

    # New primitives require a `sample` implementation, whose signature is:
    # sample(self, key: PRNGKey, *args)
    # where `PRNGKey` is the type of JAX PRNG keys.
    def sample(self, key, alpha, beta):
        v = tfd.Beta(concentration1=alpha, concentration0=beta).sample(seed=key)
        return v

    # New primitives require an implementation for their gradient strategy
    # in the `jvp_estimate` method.
    #
    # This method is called by the ADEV interpreter, and gets access
    # to primals, tangents, and two continuations for the rest of the computation.
    def jvp_estimate(self, key, primals, tangents, konts):
        kpure, kdual = konts

        # Because TFP already overloads their Beta sampler with implicit
        # differentiation rules for JVP, we directly utilize their rules.
        def _inner(alpha, beta):
            # Invoking TFP's Implicit reparametrization:
            # https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/distributions/beta.py#L292-L306
            x = tfd.Beta(concentration1=alpha, concentration0=beta).sample(seed=key)
            return x

        # We invoke JAX's JVP (which utilizes TFP's registered implicit differentiation
        # rule for Beta) to get a primal and tanget out.
        primal_out, tangent_out = jax.jvp(_inner, primals, tangents)

        # Then, we give the result to the ADEV dual'd continuation, to continue
        # ADEV's forward mode.
        return kdual((primal_out,), (tangent_out,))


# Creating an instance, to be exported and used as a sampler.
adev_beta_implicit = BetaIMPLICIT()

beta_implicit = ADEVDistribution.new(
    adev_beta_implicit, lambda v, alpha, beta: tfd.Beta(alpha, beta).log_prob(v)
)
