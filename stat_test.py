import numpy as np
from scipy import stats


def compute_significance(our_accs, baseline_accs):
    """
    Performs a Paired T-Test to evaluate the statistical significance
    between SPADE and P&A across 20 independent trials.
    """
    t_stat, p_val = stats.ttest_rel(our_accs, baseline_accs)
    print(f"Paired T-Test Results (SPADE vs P&A):")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {p_val:.4e}")
    if p_val < 0.05:
        print("Status: Statistically Significant (p < 0.05) ")
    else:
        print("Status: Not Statistically Significant (p >= 0.05) ")


def compute_one_sample_significance(our_accs, baseline_value):
    """
    Performs a One-Sample T-Test to compare SPADE's performance distribution
    against a reported fixed scalar value of DetectYSF.
    """
    t_stat, p_val = stats.ttest_1samp(our_accs, baseline_value)
    print(f"\nOne-Sample T-Test Results (SPADE vs Reported SOTA {baseline_value}):")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {p_val:.4e}")
    if p_val < 0.05:
        print("Status: Statistically Significant (p < 0.05) ")
    else:
        print("Status: Not Statistically Significant (p >= 0.05) ")


# =================================================

if __name__ == "__main__":
    print("Verifying Statistical Significance...\n")
    # Politifact(16-shot)
    politifact_spade_results = [0.867231638418079, 0.8728813559322034, 0.8615819209039548, 0.8700564971751412, 0.8700564971751412, 0.8700564971751412, 0.8615819209039548, 0.867231638418079, 0.8728813559322034, 0.867231638418079, 0.8728813559322034, 0.864406779661017, 0.867231638418079, 0.867231638418079, 0.8757062146892656, 0.867231638418079, 0.8785310734463276, 0.867231638418079, 0.8757062146892656, 0.8728813559322034]
    politifact_paa_results = [0.867231638418079, 0.847457627118644, 0.8502824858757062, 0.8559322033898306, 0.867231638418079, 0.8361581920903954, 0.8615819209039548, 0.8220338983050848, 0.7090395480225988, 0.6977401129943502, 0.8559322033898306, 0.847457627118644, 0.8559322033898306, 0.7598870056497176, 0.8502824858757062, 0.7740112994350282, 0.8502824858757062, 0.8559322033898306, 0.8502824858757062, 0.7655367231638418]
    politifact_detectysf_result = 0.8648

    # GossipCop(16-shot)
    gossipcop_spade_results = [0.8451492537313433, 0.8370646766169154, 0.8638059701492538, 0.8327114427860697, 0.8345771144278606, 0.8246268656716418, 0.8364427860696517, 0.8557213930348259, 0.8470149253731343, 0.8463930348258707, 0.8277363184079602, 0.8470149253731343, 0.8252487562189055, 0.8339552238805971, 0.822139303482587, 0.8351990049751243, 0.8401741293532339, 0.8376865671641791, 0.832089552238806, 0.8364427860696517]
    gossipcop_paa_results = [0.47077114427860695, 0.48009950248756217, 0.4925373134328358, 0.4763681592039801, 0.47077114427860695, 0.47574626865671643, 0.48009950248756217, 0.8389303482587065, 0.4900497512437811, 0.8949004975124378, 0.4732587064676617, 0.6896766169154229, 0.48445273631840796, 0.47699004975124376, 0.47761194029850745, 0.5907960199004975, 0.47077114427860695, 0.8550995024875622, 0.8669154228855721, 0.48258706467661694]
    gossipcop_detectysf_result = 0.6629

    # FANG(16-shot)
    fang_spade_results = [0.6250, 0.6194029850746269, 0.6100746268656716, 0.6063432835820896, 0.6268656716417911, 0.621268656716418, 0.5988805970149254, 0.6082089552238806, 0.6305970149253731, 0.621268656716418, 0.6044776119402985, 0.6175373134328358, 0.6138059701492538, 0.621268656716418, 0.6175373134328358, 0.625, 0.6044776119402985, 0.621268656716418, 0.6026119402985075, 0.6361940298507462]
    fang_paa_results = [0.6175373134328358, 0.585820895522388, 0.6156716417910447, 0.6324626865671642, 0.5876865671641791, 0.6194029850746269, 0.5466417910447762, 0.5746268656716418, 0.6138059701492538, 0.621268656716418, 0.6063432835820896, 0.6231343283582089, 0.6138059701492538, 0.6100746268656716, 0.5764925373134329, 0.6361940298507462, 0.6175373134328358, 0.6119402985074627, 0.582089552238806, 0.6044776119402985]
    fang_detectysf_result = 0.6371

    # Execute Paired T-Test
    compute_significance(politifact_spade_results, politifact_paa_results)
    # compute_significance(gossipcop_spade_results, gossipcop_paa_results)
    # compute_significance(fang_spade_results, fang_paa_results)

    # Execute One-Sample T-Test
    # compute_one_sample_significance(politifact_spade_results, politifact_detectysf_result)
    # compute_one_sample_significance(gossipcop_spade_results, gossipcop_detectysf_result)
    # compute_one_sample_significance(fang_spade_results, fang_detectysf_result)