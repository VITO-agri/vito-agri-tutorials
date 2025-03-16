import numpy as np
from matplotlib import pyplot as plt


def compute_anomalies(
    y: np.array,
    nyears: int,
    ref_period: int,
    label_years: list[str],
):

    # statistics over the reference period (first 4 years)
    ref = np.nanmean(np.array(y[: (36 * ref_period)]).reshape(-1, 36), axis=0)  # mean
    std = np.nanstd(
        np.array(y[: (36 * ref_period)]).reshape(-1, 36), axis=0
    )  # standard deviation
    mn = np.min(np.array(y[: (36 * ref_period)]).reshape(-1, 36), axis=0)  # minimum
    mx = np.max(np.array(y[: (36 * ref_period)]).reshape(-1, 36), axis=0)  # maximum

    diff = []
    rel_an = []
    std_an = []
    vci = []

    for i in range(nyears - ref_period):
        # absolute anomalies
        diff.append(y[(36 * (ref_period + i)) : (36 * (ref_period + i + 1))] - ref)
        # relative anomalies
        rel_an.append(
            100 * y[(36 * (ref_period + i)) : (36 * (ref_period + i + 1))] / ref
        )
        # standardized anomalies
        std_an.append(
            (y[(36 * (ref_period + i)) : (36 * (ref_period + i + 1))] - ref) / std
        )
        # vci
        vci.append(
            (y[(36 * (ref_period + i)) : (36 * (ref_period + i + 1))] - mn) / (mx - mn)
        )

    # plot the years for which anomalies are derived together with the reference
    for i in range(ref_period, nyears, 1):
        plt.plot(np.arange(1, 37), y[(i * 36) : ((i + 1) * 36)], "o-")
    plt.plot(np.arange(1, 37), ref, "-k")
    plt.legend(label_years + ["reference"])
    plt.xlabel("Dekad")  # this sets the x label
    plt.ylabel("NDVI")  # this sets the y label
    plt.show()

    # plot the absolute anomalies
    for i in range(0, len(diff)):
        plt.plot(np.arange(1, 37), diff[i], "o-")
    plt.plot(np.arange(1, 37), np.zeros(36), "-k")
    plt.legend(label_years + ["reference"])
    plt.xlabel("Dekad")  # this sets the x label
    plt.ylabel("Absolute anomaly")  # this sets the y label
    plt.show()

    # plot the relative anomalies
    for i in range(0, len(diff)):
        plt.plot(np.arange(1, 37), rel_an[i], "o-")
    plt.plot(np.arange(1, 37), np.ones(36) * 100, "-k")
    plt.legend(label_years + ["reference"])
    plt.xlabel("Dekad")  # this sets the x label
    plt.ylabel("Relative anomaly")  # this sets the y label
    plt.show()

    # plot the standardized anomalies
    for i in range(0, len(diff)):
        plt.plot(np.arange(1, 37), std_an[i], "o-")
    plt.plot(np.arange(1, 37), np.zeros(36), "-k")
    plt.legend(label_years + ["reference"])
    plt.xlabel("Dekad")  # this sets the x label
    plt.ylabel("Standardized anomaly")  # this sets the y label
    plt.show()

    # plot the VCI
    for i in range(0, len(diff)):
        plt.plot(np.arange(1, 37), vci[i], "o-")
    plt.plot(np.arange(1, 37), np.ones(36), "--k")
    plt.plot(np.arange(1, 37), np.zeros(36), "--k")
    plt.legend(label_years + ["bounds", "bounds"])
    plt.xlabel("Dekad")  # this sets the x label
    plt.ylabel("VCI")  # this sets the y label
    plt.show()

    return diff, rel_an, std_an, vci
