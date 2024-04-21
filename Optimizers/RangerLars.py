from Optimizers.Lookahead import Lookahead
from Optimizers.Ralamb import Ralamb


def RangerLars(params, alpha=0.5, k=6, *args, **kwargs):
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)


