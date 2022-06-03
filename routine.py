import math
from tqdm import tqdm
from time import time_ns
import pandas as pd
from datetime import datetime
import numpy as np

from DPriFE.lin.cipher import PublicParameter, Simluator
from DPriFE.utils.rng import create_rng


REPEAT = 1024

if __name__ == "__main__":
    rng = create_rng()
    sigma = 10
    data = []

    for log_l in range(12):

        param = PublicParameter(
            l=1 << log_l,  # input size
            m=4,  # protocol parameter
            n=4,  # protocol parameter
            p_1=10**5,  # upper bound of the inner product
            p_2=10**6,  # random parameter
            alpha=10**5,  # random parameter
            lbda=9 * 10**5,  # upper bound of the noise
        )

        for _ in tqdm(range(REPEAT)):
            point = param._asdict()

            x = rng.integers(0, math.floor(math.sqrt(param.p_1 / param.l)), size=param.l)
            y = rng.integers(0, math.floor(math.sqrt(param.p_1 / param.l)), size=param.l)
            sim = Simluator(param)
            start = time_ns()

            pk, msk = sim.setup()
            point["SETUP"] = time_ns() - start

            ct, r = sim.enc(pk, x)
            point["ENC"] = time_ns() - start

            sk = sim.key_gen(pk, msk, y, r, sigma)
            point["KEYGEN"] = time_ns() - start

            result = sim.dec(sk, ct)
            point["DEC"] = time_ns() - start

            point["expect"] = int(np.inner(x, y))
            point["result"] = int(result)
            data.append(point)

        # fig, ax = plt.subplots(figsize=(8, 5))
        # ax.hist(data, bins=int(math.sqrt(REPEAT) / 2), density=True)
        # x = np.linspace(-3 * sigma, +3 * sigma, 100)
        # ax.plot(x, stats.norm.pdf(x, 0, sigma), color="red")
        # fig.savefig(datetime.now().strftime("%Y%m%d-%H%M%S") + ".png", dpi=120)

    pd.DataFrame(data).to_csv(datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv")
