import genjax
import adevjax
import jax


# Sample a trace.
def sim(g: genjax.GenerativeFunction, args):
    key = adevjax.reap_key()  # gain access to a fresh PRNG key
    tr = g.simulate(key, args)  # sample a trace from the generative function
    return (tr.get_choices(), tr.get_score())


# Score constraints.
def density(g, chm, args):
    _, score = g.assess(
        jax.random.PRNGKey(0), chm, args
    )  # score constraints on a generative function
    return score
