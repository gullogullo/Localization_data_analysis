import math
import numpy as np
import pandas as pd
import pingouin as pg

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.power import FTestAnovaPower
from scipy.stats import ttest_ind, spearmanr, friedmanchisquare, rankdata, norm, anderson
from scipy.special import gammaln

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import scikit_posthocs as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder


def convertTuple(tup):
    stringing = ''
    for item in tup:
        stringing = stringing + str(item)
    return stringing

def most_frequent(x):
    return x.mode().iloc[0] if not x.mode().empty else None

def compute_power(pop1, pop2, alpha, num_simulations=10000):
    t_statistics = np.zeros(num_simulations)
    n1 = len(pop1)
    mean1 = np.mean(pop1)
    var1 = np.var(pop1)
    n2 = len(pop2)
    mean2 = np.mean(pop2)
    var2 = np.var(pop2)
    for i in range(num_simulations):
        sample1 = np.random.normal(mean1, np.sqrt(var1), n1)
        sample2 = np.random.normal(mean2, np.sqrt(var2), n2)
        _, t_statistic = ttest_ind(sample1, sample2, equal_var=False)
        t_statistics[i] = t_statistic
    critical_value = np.percentile(t_statistics, 100 - alpha / 2)
    power = np.mean(t_statistics > critical_value)
    return power


def test_nemenyi(data):
    nemenyi = NemenyiTestPostHoc(data)
    meanRanks, pValues = nemenyi.do()
    return meanRanks, pValues


class NemenyiTestPostHoc():

    def __init__(self, data):
        self._noOfGroups = data.shape[0]
        self._noOfSamples = data.shape[1]
        self._data = data

    def do(self):
        dataAsRanks = np.full(self._data.shape, np.nan)
        for i in range(self._noOfSamples):
            dataAsRanks[:, i] = rankdata(self._data[:, i])
        meansOfRanksOfDependentSamples = np.mean(dataAsRanks, 1)
        qValues = self._compareStatisticsOfAllPairs(meansOfRanksOfDependentSamples)
        pValues = self._calculatePValues(qValues)

        return qValues, pValues

    def _compareStatisticsOfAllPairs(self, meansOfRanks):
        noOfMeansOfRanks = len(meansOfRanks)
        compareResults = np.zeros((noOfMeansOfRanks-1, noOfMeansOfRanks))
        for i in range(noOfMeansOfRanks-1):
            for j in range(i+1, noOfMeansOfRanks):
                compareResults[i][j] = self._compareStatisticsOfSinglePair((meansOfRanks[i], meansOfRanks[j]))
        return compareResults

    def _compareStatisticsOfSinglePair(self, meansOfRanksPair):
        diff = abs(meansOfRanksPair[0] - meansOfRanksPair[1])
        qval = diff / np.sqrt(self._noOfGroups * (self._noOfGroups + 1) / (6 * self._noOfSamples))
        return qval * np.sqrt(2)

    def _calculatePValues(self, qValues):
        for qRow in qValues:
            for i in range(len(qRow)):
                qRow[i] = self._ptukey(qRow[i], 1, self._noOfGroups, np.inf)
        return 1 - qValues

    def _wprob(self, w, rr, cc):
        nleg = 12
        ihalf = 6

        C1 = -30
        C2 = -50
        C3 = 60
        M_1_SQRT_2PI = 1 / np.sqrt(2 * np.pi)
        bb = 8
        wlar = 3
        wincr1 = 2
        wincr2 = 3
        xleg = [
            0.981560634246719250690549090149,
            0.904117256370474856678465866119,
            0.769902674194304687036893833213,
            0.587317954286617447296702418941,
            0.367831498998180193752691536644,
            0.125233408511468915472441369464
        ]
        aleg = [
            0.047175336386511827194615961485,
            0.106939325995318430960254718194,
            0.160078328543346226334652529543,
            0.203167426723065921749064455810,
            0.233492536538354808760849898925,
            0.249147045813402785000562436043
        ]

        qsqz = w * 0.5

        if qsqz >= bb:
            return 1.0

        # find (f(w/2) - 1) ^ cc
        # (first term in integral of hartley's form).

        pr_w = 2 * norm.cdf(qsqz) - 1
        if pr_w >= np.exp(C2 / cc):
            pr_w = pr_w ** cc
        else:
            pr_w = 0.0

        # if w is large then the second component of the
        # integral is small, so fewer intervals are needed.

        wincr = wincr1 if w > wlar else wincr2

        # find the integral of second term of hartley's form
        # for the integral of the range for equal-length
        # intervals using legendre quadrature.  limits of
        # integration are from (w/2, 8).  two or three
        # equal-length intervals are used.

        # blb and bub are lower and upper limits of integration.

        blb = qsqz
        binc = (bb - qsqz) / wincr
        bub = blb + binc
        einsum = 0.0

        # integrate over each interval

        cc1 = cc - 1.0
        for wi in range(1, wincr + 1):
            elsum = 0.0
            a = 0.5 * (bub + blb)

            # legendre quadrature with order = nleg

            b = 0.5 * (bub - blb)

            for jj in range(1, nleg + 1):
                if (ihalf < jj):
                    j = (nleg - jj) + 1
                    xx = xleg[j-1]
                else:
                    j = jj
                    xx = -xleg[j-1]
                c = b * xx
                ac = a + c

                # if exp(-qexpo/2) < 9e-14
                # then doesn't contribute to integral

                qexpo = ac * ac
                if qexpo > C3:
                    break

                pplus = 2 * norm.cdf(ac)
                pminus = 2 * norm.cdf(ac, w)

                # if rinsum ^ (cc-1) < 9e-14, */
                # then doesn't contribute to integral */

                rinsum = (pplus * 0.5) - (pminus * 0.5)
                if (rinsum >= np.exp(C1 / cc1)):
                    rinsum = (aleg[j-1] * np.exp(-(0.5 * qexpo))) * (rinsum ** cc1)
                    elsum += rinsum

            elsum *= (((2.0 * b) * cc) * M_1_SQRT_2PI)
            einsum += elsum
            blb = bub
            bub += binc

        # if pr_w ^ rr < 9e-14, then return 0
        pr_w += einsum
        if pr_w <= np.exp(C1 / rr):
            return 0

        pr_w = pr_w ** rr
        if (pr_w >= 1):
            return 1
        return pr_w

    def _ptukey(self, q, rr, cc, df):

        M_LN2 = 0.69314718055994530942

        nlegq = 16
        ihalfq = 8

        eps1 = -30.0
        eps2 = 1.0e-14
        dhaf = 100.0
        dquar = 800.0
        deigh = 5000.0
        dlarg = 25000.0
        ulen1 = 1.0
        ulen2 = 0.5
        ulen3 = 0.25
        ulen4 = 0.125
        xlegq = [
            0.989400934991649932596154173450,
            0.944575023073232576077988415535,
            0.865631202387831743880467897712,
            0.755404408355003033895101194847,
            0.617876244402643748446671764049,
            0.458016777657227386342419442984,
            0.281603550779258913230460501460,
            0.950125098376374401853193354250e-1
        ]
        alegq = [
            0.271524594117540948517805724560e-1,
            0.622535239386478928628438369944e-1,
            0.951585116824927848099251076022e-1,
            0.124628971255533872052476282192,
            0.149595988816576732081501730547,
            0.169156519395002538189312079030,
            0.182603415044923588866763667969,
            0.189450610455068496285396723208
        ]

        if q <= 0:
            return 0

        if (df < 2) or (rr < 1) or (cc < 2):
            return float('nan')

        if np.isfinite(q) is False:
            return 1

        if df > dlarg:
            return self._wprob(q, rr, cc)

        # in fact we don't need the code below and majority of variables:

        # calculate leading constant

        f2 = df * 0.5
        f2lf = ((f2 * np.log(df)) - (df * M_LN2)) - gammaln(f2)
        f21 = f2 - 1.0

        # integral is divided into unit, half-unit, quarter-unit, or
        # eighth-unit length intervals depending on the value of the
        # degrees of freedom.

        ff4 = df * 0.25
        if df <= dhaf:
            ulen = ulen1
        elif df <= dquar:
            ulen = ulen2
        elif df <= deigh:
            ulen = ulen3
        else:
            ulen = ulen4

        f2lf += np.log(ulen)

        ans = 0.0

        for i in range(1, 51):
            otsum = 0.0

            # legendre quadrature with order = nlegq
            # nodes (stored in xlegq) are symmetric around zero.

            twa1 = (2*i - 1) * ulen

            for jj in range(1, nlegq + 1):
                if (ihalfq < jj):
                    j = jj - ihalfq - 1
                    t1 = (f2lf + (f21 * np.log(twa1 + (xlegq[j] * ulen)))) - (((xlegq[j] * ulen) + twa1) * ff4)
                else:
                    j = jj - 1
                    t1 = (f2lf + (f21 * np.log(twa1 - (xlegq[j] * ulen)))) + (((xlegq[j] * ulen) - twa1) * ff4)

                # if exp(t1) < 9e-14, then doesn't contribute to integral
                if t1 >= eps1:
                    if ihalfq < jj:
                        qsqz = q * np.sqrt(((xlegq[j] * ulen) + twa1) * 0.5)
                    else:
                        qsqz = q * np.sqrt(((-(xlegq[j] * ulen)) + twa1) * 0.5)

                    wprb = self._wprob(qsqz, rr, cc)
                    rotsum = (wprb * alegq[j]) * np.exp(t1)
                    otsum += rotsum

            # if integral for interval i < 1e-14, then stop.
            # However, in order to avoid small area under left tail,
            # at least  1 / ulen  intervals are calculated.

            if (i * ulen >= 1.0) and (otsum <= eps2):
                break

            ans += otsum

        return min(1, ans)

# FLAG FOR PLOTS
want_figures = False
want_qq = False
want_xy_plots = False
# FLAG FOR MIXED MODELS
want_lm = False
# FLAG FOR CORRELATION
want_correlation = False
# FLAG FOR STATS
want_stats = False


mpl.rcParams['figure.dpi'] = 300
#plt.rcParams.update({'font.size': 22})
dataset = pd.read_csv('datasets/complete_bi_NH.csv')
#dataset = pd.read_csv('datasets/complete_bi_NH_meno_il_primo.csv')
clean_dataset = pd.DataFrame()

variables = ['Signed_error', 'Unsigned_error', 'Head_rotation', 'Head_distance']
var_names = ['Signed error', 'Unsigned error', 'Head rotation', 'Head distance']
individual_vars = ['Age', 'MacroCause', 'Experience DX', 'Experience SX',
       'Threshold w/o 500 DX', 'Threshold w/o 1000 DX',
       'Threshold w/o 2000 DX', 'Threshold w/o 4000 DX',
       'Threshold w/o 500 SX', 'Threshold w/o 1000 SX',
       'Threshold w/o 2000 SX', 'Threshold w/o 4000 SX', 'Threshold w/ 500 DX',
       'Threshold w/ 1000 DX', 'Threshold w/ 2000 DX', 'Threshold w/ 4000 DX',
       'Threshold w/ 500 SX', 'Threshold w/ 1000 SX', 'Threshold w/ 2000 SX',
       'Threshold w/ 4000 SX']

# REMOVE OUTLIERS WRT UNSIGNED ERROR
conditions = ['ICSX_NOICDX', 'ICSX_ICDX', 'NOICSX_ICDX', 'PASX_NOPADX', 'PASX_PADX',
              'NOPASX_NOPADX', 'NOPASX_PADX']
results_list = []
for condition in conditions:
    one_condition_dataset = dataset[dataset['Condition'] == condition]
    condition_participants = one_condition_dataset['Participant'].unique()
    count_participants = 0
    for participant in condition_participants:
        one_condition_one_participant = one_condition_dataset[one_condition_dataset['Participant'] == participant]
        if len(one_condition_one_participant) != 65:
            count_participants += 1
            result = {'Condition': condition,
                      'Participant': participant,
                      'Number of answers': len(one_condition_one_participant)}
            results_list.append(result)
    mean_unsigned_error = one_condition_dataset['Unsigned_error'].mean()
    std_unsigned_error = one_condition_dataset['Unsigned_error'].std()
    threshold = mean_unsigned_error + 3 * std_unsigned_error
    no_outliers_dataset = one_condition_dataset[one_condition_dataset['Unsigned_error'] <= threshold]
    result = {'Condition': condition,
              'Original Number of participants': len(one_condition_dataset['Participant'].unique()),
              'Original Number of answers': len(one_condition_dataset),
              'Number of participants not complete': count_participants,
              'Outliers threshold': threshold,
              'Filtered Number of participants': len(no_outliers_dataset['Participant'].unique()),
              'Filtered Number of answers': len(no_outliers_dataset),
              'Percentage outliers': (len(one_condition_dataset) - len(
                  no_outliers_dataset)) / len(one_condition_dataset) * 100}
    results_list.append(result)
    clean_dataset = pd.concat([clean_dataset, no_outliers_dataset], ignore_index=True)
results_df = pd.DataFrame(results_list)
results_df.to_csv('filtering_results.csv', index=False)


print("Original Dataset Shape:", len(dataset))
print("Filtered Dataset Shape:", len(clean_dataset))
for individual_var in individual_vars:
    #print("Individual variable: ", individual_var)
    levels_individual_var = clean_dataset[individual_var].unique()
    #print('Levels: ', len(levels_individual_var))
    # Filter dataset for each individual variable level
    for var_level in levels_individual_var:
        #print('Level: ', var_level)
        one_level_dataset = clean_dataset[clean_dataset[individual_var] == var_level]
    if individual_var == 'Age':
        ages = []
        for participant in clean_dataset['Participant'].unique():
            ages.append(clean_dataset[clean_dataset['Participant'] == participant][individual_var].unique())
        print(np.mean(ages))
        print(np.std(ages))


# LOG TRANSFORM ALL DATA
min_values = []
min_values_IC = []
min_values_PA = []
for var in variables:
    print('VARIABLE', var)
    IC_data = clean_dataset[(clean_dataset['Condition'] == 'NOICSX_ICDX') | (
        clean_dataset['Condition'] == 'ICSX_ICDX') | (
            clean_dataset['Condition'] == 'ICSX_NOICDX')]
    PA_data = clean_dataset[(clean_dataset['Condition'] == 'NOPASX_PADX') | (
        clean_dataset['Condition'] == 'PASX_PADX') | (
            clean_dataset['Condition'] == 'NOPASX_NOPADX') | (
                clean_dataset['Condition'] == 'PASX_NOPADX')]
    min_values.append(min(clean_dataset[var]))
    print('min_value', min_values[-1])
    min_values_IC.append(min(IC_data[var]))
    min_values_PA.append(min(PA_data[var]))
    shifted_data = [x - min_values[-1] + 1 for x in clean_dataset[var]]
    log_data = [np.log(x) for x in shifted_data]
    clean_dataset[var] = log_data


# INTRODUCE POPULATION COLUMN: HA or CI 
clean_dataset['Group'] = clean_dataset['Group'].str.contains('CI', case=False, regex=True)
clean_dataset['Group'] = clean_dataset['Group'].map({True: 'CI', False: 'HA'})

# RENAME COLUMNS
clean_dataset.rename(columns={'Threshold w/o 500 DX': 'Threshold_wo_500_DX', 
                              'Threshold w/o 1000 DX': 'Threshold_wo_1000_DX', 
                              'Threshold w/o 2000 DX': 'Threshold_wo_2000_DX', 
                              'Threshold w/o 4000 DX': 'Threshold_wo_4000_DX', 
                              'Threshold w/o 500 SX': 'Threshold_wo_500_SX', 
                              'Threshold w/o 1000 SX': 'Threshold_wo_1000_SX', 
                              'Threshold w/o 2000 SX': 'Threshold_wo_2000_SX', 
                              'Threshold w/o 4000 SX': 'Threshold_wo_4000_SX', 
                              'Threshold w/ 500 DX': 'Threshold_w_500_DX', 
                              'Threshold w/ 1000 DX': 'Threshold_w_1000_DX', 
                              'Threshold w/ 2000 DX': 'Threshold_w_2000_DX', 
                              'Threshold w/ 4000 DX': 'Threshold_w_4000_DX', 
                              'Threshold w/ 500 SX': 'Threshold_w_500_SX', 
                              'Threshold w/ 1000 SX': 'Threshold_w_1000_SX', 
                              'Threshold w/ 2000 SX': 'Threshold_w_2000_SX', 
                              'Threshold w/ 4000 SX': 'Threshold_w_4000_SX', 
                              'Experience DX': 'Experience_DX', 'Experience SX': 'Experience_SX'}, inplace=True)               


# AGGREGATE DATA OF TARGETS WITH MEDIANS
clean_dataset = clean_dataset.groupby(['Target', 'Participant', 'Condition'], as_index=False).agg(
        {'Signed_error': 'median', 'Unsigned_error': 'median', 'Head_rotation': 'median', 'Head_distance': 'median',
         'Age': 'median', 'Experience_DX': 'median', 'Experience_SX': 'median',
         'Threshold_wo_500_DX': 'median', 'Threshold_wo_1000_DX': 'median', 'Threshold_wo_2000_DX': 'median', 
         'Threshold_wo_4000_DX': 'median', 'Threshold_wo_500_SX': 'median', 'Threshold_wo_1000_SX': 'median', 
         'Threshold_wo_2000_SX': 'median', 'Threshold_wo_4000_SX': 'median', 'Threshold_w_500_DX': 'median', 
         'Threshold_w_1000_DX': 'median', 'Threshold_w_2000_DX': 'median', 'Threshold_w_4000_DX': 'median',
         'Threshold_w_500_SX': 'median', 'Threshold_w_1000_SX': 'median', 'Threshold_w_2000_SX': 'median', 
         'Threshold_w_4000_SX': 'median', 'MacroCause': most_frequent})


participants = clean_dataset['Participant'].unique()
data_medians = {'Condition': [], 'Variable': [], 'Participant': [], 'Median': [], 'Target': [], 
        'Age': [], 'MacroCause': [], 'Experience_DX': [], 'Experience_SX': [],
       'Threshold_wo_500_DX': [], 'Threshold_wo_1000_DX': [],
       'Threshold_wo_2000_DX': [], 'Threshold_wo_4000_DX': [],
       'Threshold_wo_500_SX': [], 'Threshold_wo_1000_SX': [],
       'Threshold_wo_2000_SX': [], 'Threshold_wo_4000_SX': [], 'Threshold_w_500_DX': [],
       'Threshold_w_1000_DX': [], 'Threshold_w_2000_DX': [], 'Threshold_w_4000_DX': [],
       'Threshold_w_500_SX': [], 'Threshold_w_1000_SX': [], 'Threshold_w_2000_SX': [],
       'Threshold_w_4000_SX': []}
dataset_medians = pd.DataFrame(data_medians)

for participant in participants:
    one_participant_dataset = clean_dataset[clean_dataset['Participant'] == participant]
    for n, var in enumerate(variables):
        for m, condition in enumerate(one_participant_dataset['Condition'].unique()):
            one_condition_one_participant = one_participant_dataset[
            one_participant_dataset['Condition'] == condition]
            right_half_alpha = np.asarray([np.mean(one_condition_one_participant['Threshold_wo_500_DX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_wo_1000_DX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_wo_2000_DX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_wo_4000_DX'].unique())])
            left_half_alpha = np.asarray([np.mean(one_condition_one_participant['Threshold_wo_500_SX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_wo_1000_SX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_wo_2000_SX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_wo_4000_SX'].unique())])
            cosine_sim = cosine_similarity(right_half_alpha.reshape(1, -1), left_half_alpha.reshape(1, -1))
            cosine_dist_without = np.round(1 - cosine_sim[0][0], 4)
            right_half_with = np.asarray([np.mean(one_condition_one_participant['Threshold_w_500_DX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_w_1000_DX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_w_2000_DX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_w_4000_DX'].unique())])
            left_half_with = np.asarray([np.mean(one_condition_one_participant['Threshold_w_500_SX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_w_1000_SX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_w_2000_SX'].unique()),
                                        np.mean(one_condition_one_participant['Threshold_w_4000_SX'].unique())])
            cosine_sim = cosine_similarity(right_half_with.reshape(1, -1), left_half_with.reshape(1, -1))
            cosine_dist_with = np.round(1 - cosine_sim[0][0], 4)
            for target in one_condition_one_participant['Target'].unique():
                one_participant_one_target = one_condition_one_participant[
                    one_condition_one_participant['Target'] == target]
                median_var = np.median(one_participant_one_target[var])
                new_row = {'Condition': [condition], 'Variable': [var], 'Participant': [participant], 
                           'Median': [median_var], 'Target': [target], 
                           'Age': [np.mean(one_participant_one_target['Age'].unique())], 
                            'MacroCause': [one_participant_one_target['MacroCause'].unique()[0]], 
                            'Experience_DX': [np.mean(one_participant_one_target['Experience_DX'].unique())], 
                            'Experience_SX': [np.mean(one_participant_one_target['Experience_SX'].unique())], 
                            'Threshold_wo_500_DX': [np.mean(one_participant_one_target['Threshold_wo_500_DX'].unique())], 
                            'Threshold_wo_1000_DX': [np.mean(one_participant_one_target['Threshold_wo_1000_DX'].unique())], 
                            'Threshold_wo_2000_DX': [np.mean(one_participant_one_target['Threshold_wo_2000_DX'].unique())],  
                            'Threshold_wo_4000_DX': [np.mean(one_participant_one_target['Threshold_wo_4000_DX'].unique())], 
                            'Threshold_wo_500_SX': [np.mean(one_participant_one_target['Threshold_wo_500_SX'].unique())], 
                            'Threshold_wo_1000_SX': [np.mean(one_participant_one_target['Threshold_wo_1000_SX'].unique())], 
                            'Threshold_wo_2000_SX': [np.mean(one_participant_one_target['Threshold_wo_2000_SX'].unique())], 
                            'Threshold_wo_4000_SX': [np.mean(one_participant_one_target['Threshold_wo_4000_SX'].unique())], 
                            'Threshold_w_500_DX': [np.mean(one_participant_one_target['Threshold_w_500_DX'].unique())], 
                            'Threshold_w_1000_DX': [np.mean(one_participant_one_target['Threshold_w_1000_DX'].unique())], 
                            'Threshold_w_2000_DX': [np.mean(one_participant_one_target['Threshold_w_2000_DX'].unique())],  
                            'Threshold_w_4000_DX': [np.mean(one_participant_one_target['Threshold_w_4000_DX'].unique())], 
                            'Threshold_w_500_SX': [np.mean(one_participant_one_target['Threshold_w_500_SX'].unique())], 
                            'Threshold_w_1000_SX': [np.mean(one_participant_one_target['Threshold_w_1000_SX'].unique())], 
                            'Threshold_w_2000_SX': [np.mean(one_participant_one_target['Threshold_w_2000_SX'].unique())], 
                            'Threshold_w_4000_SX': [np.mean(one_participant_one_target['Threshold_w_4000_SX'].unique())]} 
                new_row = pd.DataFrame(new_row)
                dataset_medians = pd.concat([dataset_medians, new_row], ignore_index=True)
  

# CORRELATION ANALYSIS PER CONDITION PER PARTICIPANT
results_list = []
for condition in conditions:
    heatmap_data_condition = clean_dataset[clean_dataset['Condition'] == condition]
    if want_xy_plots:
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Head_distance'], heatmap_data_condition['Signed_error'])
        fig.savefig(condition + '_HD_SE.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Head_distance'], heatmap_data_condition['Unsigned_error'])
        fig.savefig(condition + '_HD_UE.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Head_rotation'], heatmap_data_condition['Signed_error'])
        fig.savefig(condition + '_HR_SE.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Head_rotation'], heatmap_data_condition['Unsigned_error'])
        fig.savefig(condition + '_HR_UE.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Head_rotation'], heatmap_data_condition['Head_distance'])
        fig.savefig(condition + '_HR_HD.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Age'], heatmap_data_condition['Unsigned_error'])
        fig.savefig(condition + '_Age_UE.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Age'], heatmap_data_condition['Head_distance'])
        fig.savefig(condition + '_Age_HD.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Age'], heatmap_data_condition['Head_rotation'])
        fig.savefig(condition + '_Age_HR.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Age'], heatmap_data_condition['Signed_error'])
        fig.savefig(condition + '_Age_SE.png')
        plt.close()
        fig, ax = plt.subplots()
        ax.scatter(heatmap_data_condition['Unsigned_error'], heatmap_data_condition['Signed_error'])
        fig.savefig(condition + '_UE_SE.png')
        plt.close()
    correlation_coefficient, p_value = stats.pearsonr(heatmap_data_condition['Signed_error'],
                                                      heatmap_data_condition['Head_rotation'])
    if p_value < 0.05:
        result = {'Condition': condition,
                  'Test': 'Pearson Signed error - Head rotation',
                  'Correlation Coefficient': correlation_coefficient,
                  'P-value': p_value}
        results_list.append(result)
    correlation_coefficient, p_value = stats.pearsonr(heatmap_data_condition['Unsigned_error'],
                                                      heatmap_data_condition['Head_distance'])
    if p_value < 0.05:
        result = {'Condition': condition,
                  'Test': 'Pearson Unsigned error - Head distance',
                  'Correlation Coefficient': correlation_coefficient,
                  'P-value': p_value}
        results_list.append(result)
    correlation_coefficient, p_value = stats.pearsonr(heatmap_data_condition['Unsigned_error'],
                                                      heatmap_data_condition['Experience_DX'])
    if p_value < 0.05:
        result = {'Condition': condition,
                    'Test': 'Pearson Unsigned error - Experience_DX',
                    'Correlation Coefficient': correlation_coefficient,
                    'P-value': p_value}
        results_list.append(result)
    correlation_coefficient, p_value = stats.pearsonr(heatmap_data_condition['Unsigned_error'],
                                                      heatmap_data_condition['Experience_SX'])
    if p_value < 0.05:
        result = {'Condition': condition,
                    'Test': 'Pearson Unsigned error - Experience_SX',
                    'Correlation Coefficient': correlation_coefficient,
                    'P-value': p_value}
        results_list.append(result)
    correlation_coefficient, p_value = stats.pearsonr(heatmap_data_condition['Unsigned_error'],
                                                      heatmap_data_condition['Age'])
    if p_value < 0.05:
        result = {'Condition': condition,
                    'Test': 'Pearson Unsigned error - Age',
                    'Correlation Coefficient': correlation_coefficient,
                    'P-value': p_value}
        results_list.append(result)
    for participant in heatmap_data_condition['Participant'].unique():
        heatmap_target = heatmap_data_condition[heatmap_data_condition['Participant'] == participant]
        correlation_coefficient, p_value = stats.pearsonr(heatmap_target['Signed_error'],
                                                          heatmap_target['Head_rotation'])
        if p_value < 0.05:
            result = {'Condition': condition,
                      'Participant': participant,
                      'Test': 'Pearson Signed error - Head rotation',
                      'Correlation Coefficient': correlation_coefficient,
                      'P-value': p_value}
            results_list.append(result)
        correlation_coefficient, p_value = stats.pearsonr(heatmap_target['Unsigned_error'],
                                                          heatmap_target['Head_distance'])
        if p_value < 0.05:
            result = {'Condition': condition,
                      'Participant': participant,
                      'Test': 'Pearson Unsigned error - Head distance',
                      'Correlation Coefficient': correlation_coefficient,
                      'P-value': p_value}
            results_list.append(result)
    if condition in ['NOPASX_PADX', 'PASX_NOPADX']:
        for participant in heatmap_data_condition['Participant'].unique():
            thresh_4k_L_wo = np.mean(heatmap_data_condition[
                heatmap_data_condition['Participant'] == participant]['Threshold_wo_4000_SX'])
            thresh_4k_L_w = np.mean(heatmap_data_condition[
                heatmap_data_condition['Participant'] == participant]['Threshold_w_4000_SX'])
            thresh_4k_R_wo = np.mean(heatmap_data_condition[
                heatmap_data_condition['Participant'] == participant]['Threshold_wo_4000_DX'])
            thresh_4k_R_w = np.mean(heatmap_data_condition[
                heatmap_data_condition['Participant'] == participant]['Threshold_w_4000_DX'])
            error = np.exp(heatmap_data_condition[
                    heatmap_data_condition['Participant'] == participant]['Unsigned_error'] + min_values[n] - 1)
            #print(condition)
            #print('Subject')
            #print(participant)
            #print('Thresholds')
            #print(thresh_4k_L_wo)
            #print(thresh_4k_L_w)
            #print(thresh_4k_R_wo)
            #print(thresh_4k_R_w)
            #print('Median error')
            #print(np.median(error))

results_df = pd.DataFrame(results_list)
results_df.to_csv('correlation_results.csv', index=False)


# LINEAR MIXED MODELS ANALYSIS
if want_lm:
    label_encoder = LabelEncoder()
    #clean_dataset['Condition'] = label_encoder.fit_transform(clean_dataset['Condition'])
    #clean_dataset['MacroCause'] = label_encoder.fit_transform(clean_dataset['MacroCause'])
    #clean_dataset['Group'] = label_encoder.fit_transform(clean_dataset['Group'])

    individual_vars = ['Age', 'MacroCause', 'Experience_DX', 'Experience_SX',
        'Threshold_wo_500_DX', 'Threshold_wo_1000_DX', 'Threshold_wo_2000_DX', 'Threshold_wo_4000_DX',
        'Threshold_wo_500_SX', 'Threshold_wo_1000_SX', 'Threshold_wo_2000_SX', 'Threshold_wo_4000_SX',
        'Threshold_w_500_DX', 'Threshold_w_1000_DX', 'Threshold_w_2000_DX', 'Threshold_w_4000_DX',
        'Threshold_w_500_SX', 'Threshold_w_1000_SX', 'Threshold_w_2000_SX', 'Threshold_w_4000_SX'] 


    combined_results_list = []
    mixed_dataset = clean_dataset
    mixed_dataset['condition1'] = mixed_dataset.apply(lambda row: row['Condition'] if row['Group'] == 'CI' else 0, axis=1)
    mixed_dataset['condition2'] = mixed_dataset.apply(lambda row: row['Condition'] if row['Group'] == 'HA' else 0, axis=1)
    mixed_dataset['MacroCause'] = label_encoder.fit_transform(mixed_dataset['MacroCause'])
    mixed_dataset['condition1'] = mixed_dataset['condition1'].astype('category')
    mixed_dataset['condition2'] = mixed_dataset['condition2'].astype('category')
    mixed_dataset['Group'] = mixed_dataset['Group'].astype('category')
    for var in variables:
        formula = f"{var} ~ Group * Condition" 
        title = formula
        try:      
            model = smf.mixedlm(formula, data=mixed_dataset, groups=mixed_dataset['Participant'])
            result = model.fit()
            summary_df = result.summary().tables[1].to_html()
            summary_df = pd.read_html(summary_df, header=0, index_col=0)[0]
            summary_df['Title'] = title
            significant_rows = summary_df[abs(summary_df['P>|z|']) < 0.05]
            combined_results_list.append(significant_rows)
        except np.linalg.LinAlgError as e:
            print(f"Error for {title}: {e}")
        for individual_var in individual_vars:
            formula = f"{var} ~ Group * condition1 * condition2 + {individual_var}" 
            title = formula
            #formula = f"{var} ~ Group + Condition + Target + {individual_var}" 
            #formula = f"{var} ~ Condition + Target + Age + MacroCause + Experience_DX + Experience_SX + Threshold_wo_500_DX + Threshold_wo_1000_DX + Threshold_wo_2000_DX + Threshold_wo_4000_DX + Threshold_wo_500_SX + Threshold_wo_1000_SX + Threshold_wo_2000_SX + Threshold_wo_4000_SX + Threshold_w_500_DX + Threshold_w_1000_DX + Threshold_w_2000_DX + Threshold_w_4000_DX + Threshold_w_500_SX + Threshold_w_1000_SX + Threshold_w_2000_SX + Threshold_w_4000_SX"  
            try:      
                model = sm.MixedLM.from_formula(formula, 
                                                data=mixed_dataset, 
                                                groups=mixed_dataset['Participant'])
                result = model.fit(method='nm', maxiter=1000)
                summary_df = result.summary().tables[1].to_html()
                summary_df = pd.read_html(summary_df, header=0, index_col=0)[0]
                summary_df['Title'] = title
                significant_rows = summary_df[abs(summary_df['P>|z|']) < 0.05]
                combined_results_list.append(significant_rows)
            except np.linalg.LinAlgError as e:
                print(f"Error for {title}: {e}")


    mixed_results = pd.concat(combined_results_list)
    mixed_results.to_csv(f'mixed_results.csv')


# GENERAL LISTS FOR PLOTS, BOXPLOTS, AND STATISTICS
targets = [-90., -75., -60., -45., -30., -15., 0., 15., 30., 45., 60., 75., 90.]
conditions_ON = ['PASX_PADX', 'ICSX_ICDX']
conditions_L = ['PASX_NOPADX', 'ICSX_NOICDX']
conditions_R = ['NOPASX_PADX', 'NOICSX_ICDX']
conditions_CI = ['ICSX_NOICDX', 'ICSX_ICDX', 'NOICSX_ICDX']
conditions_HA = ['PASX_NOPADX', 'PASX_PADX', 'NOPASX_NOPADX', 'NOPASX_PADX']  
colors_ON = ['y', 'g']
colors_all = ['b', 'y', 'r', 'c', 'g', 'grey', 'm']
colors_plot = colors_all * len(targets)
y_labels = ['Signed error [°]', 'Unsigned error [°]', 'Head rotation [°]', 'Head distance [m]']
titles = ['CI L', 'CI ON', 'CI R', 'HA L', 'HA ON', 'NO HA', 'HA R']
patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


# CORRELATION ANALYSIS
if want_correlation:
    for nn, condition in enumerate(conditions):
        heatmap_data_condition = clean_dataset[(clean_dataset['Condition'] == condition)].copy()
        heatmap_data_condition.rename(columns={'Signed_error': 'Signed error', 'Head_rotation': 'Head rotation',
                                            'Unsigned_error': 'Unsigned error',  'Head_distance': 'Head distance',
                                                'Experience_DX': 'Experience R', 'Experience_SX': 'Experience L',
                                                'Threshold_w_500_DX': 'T500 w/ R', 'Threshold_w_1000_DX': 'T1000 w/ R',
                                                'Threshold_w_2000_DX': 'T2000 w/ R', 'Threshold_w_4000_DX': 'T4000 w/ R',
                                                'Threshold_w_500_SX': 'T500 w/ L', 'Threshold_w_1000_SX': 'T1000 w/ L',
                                                'Threshold_w_2000_SX': 'T2000 w/ L', 'Threshold_w_4000_SX': 'T4000 w/ L',
                                                'Threshold_wo_500_DX': 'T500 w/o R', 'Threshold_wo_1000_DX': 'T1000 w/o R',
                                                'Threshold_wo_2000_DX': 'T2000 w/o R', 'Threshold_wo_4000_DX': 'T4000 w/o R',
                                                'Threshold_wo_500_SX': 'T500 w/o L', 'Threshold_wo_1000_SX': 'T1000 w/o L',
                                                'Threshold_wo_2000_SX': 'T2000 w/o L', 'Threshold_wo_4000_SX': 'T4000 w/o L'}, 
                                                inplace=True)
        for thr in ['T500 w/ R', 'T1000 w/ R', 'T2000 w/ R', 'T4000 w/ R', 
                    'T500 w/ L', 'T1000 w/ L', 'T2000 w/ L', 'T4000 w/ L',
                    'T500 w/o R', 'T1000 w/o R', 'T2000 w/o R', 'T4000 w/o R',
                    'T500 w/o L', 'T1000 w/o L', 'T2000 w/o L', 'T4000 w/o L',
                    'Experience R', 'Experience L']:
            min_value = heatmap_data_condition[thr].min()
            max_value = heatmap_data_condition[thr].max()

            heatmap_data_condition[thr] = (heatmap_data_condition[thr] - min_value) / (max_value - min_value)

        # Calculate median for each combination of 'Target' and 'Participant'
        right_half_alpha = np.asarray([np.mean(one_condition_one_participant['Threshold_wo_500_DX'].unique()),
                                    np.mean(one_condition_one_participant['Threshold_wo_1000_DX'].unique()),
                                    np.mean(one_condition_one_participant['Threshold_wo_2000_DX'].unique()),
                                    np.mean(one_condition_one_participant['Threshold_wo_4000_DX'].unique())])
        left_half_alpha = np.asarray([np.mean(one_condition_one_participant['Threshold_wo_500_SX'].unique()),
                                    np.mean(one_condition_one_participant['Threshold_wo_1000_SX'].unique()),
                                    np.mean(one_condition_one_participant['Threshold_wo_2000_SX'].unique()),
                                    np.mean(one_condition_one_participant['Threshold_wo_4000_SX'].unique())])
        cosine_sim = cosine_similarity(right_half_alpha.reshape(1, -1), left_half_alpha.reshape(1, -1))
        cosine_dist_without = np.round(1 - cosine_sim[0][0], 4)
        grouped_df = heatmap_data_condition.groupby(['Target', 'Participant'], as_index=False).agg(
            {'Signed error': 'median', 'Unsigned error': 'median', 'Head rotation': 'median', 'Head distance': 'median',
            'Condition': most_frequent, 'Age': 'median', 'Experience R': 'median', 'Experience L': 'median',
            'T500 w/o R': 'median', 'T1000 w/o R': 'median', 'T2000 w/o R': 'median', 'T4000 w/o R': 'median',
            'T500 w/o L': 'median', 'T1000 w/o L': 'median', 'T2000 w/o L': 'median', 'T4000 w/o L': 'median',
            'T500 w/ R': 'median', 'T1000 w/ R': 'median', 'T2000 w/ R': 'median', 'T4000 w/ R': 'median',
            'T500 w/ L': 'median', 'T1000 w/ L': 'median', 'T2000 w/ L': 'median', 'T4000 w/ L': 'median'})
        #if condition in conditions_CI:
        #    grouped_df.drop(columns=['Participant',  'Target', 'Condition', 'T500 w/o L', 'T1000 w/o L', 'T2000 w/o L', 'T4000 w/o L', 'T500 w/o R', 'T1000 w/o R', 'T2000 w/o R', 'T4000 w/o R'], inplace=True)
        #    grouped_df = grouped_df[['Unsigned error', 'Head distance', 'Signed error', 'Head rotation', 'Age', 'Experience L', 'Experience R', 'T500 w/ L', 'T1000 w/ L', 'T2000 w/ L', 'T4000 w/ L', 'T500 w/ R', 'T1000 w/ R', 'T2000 w/ R', 'T4000 w/ R']]
        #else:
        #    grouped_df.drop(columns=['Participant',  'Target', 'Condition'], inplace=True)
        #    grouped_df = grouped_df[['Unsigned error', 'Head distance', 'Signed error', 'Head rotation', 'Age', 'Experience L', 'Experience R', 'T500 w/o L', 'T1000 w/o L', 'T2000 w/o L', 'T4000 w/o L', 'T500 w/ L', 'T1000 w/ L', 'T2000 w/ L', 'T4000 w/ L', 'T500 w/o R', 'T1000 w/o R', 'T2000 w/o R', 'T4000 w/o R', 'T500 w/ R', 'T1000 w/ R', 'T2000 w/ R', 'T4000 w/ R']]
        grouped_df.drop(columns=['Participant',  'Target', 'Condition', 
                                 'T500 w/ R', 'T1000 w/ R', 'T2000 w/ R', 'T4000 w/ R', 
                                 'T500 w/ L', 'T1000 w/ L', 'T2000 w/ L', 'T4000 w/ L',
                                 'T500 w/o R', 'T1000 w/o R', 'T2000 w/o R', 'T4000 w/o R',
                                 'T500 w/o L', 'T1000 w/o L', 'T2000 w/o L', 'T4000 w/o L',
                                 'Experience R', 'Experience L'], inplace=True)
        grouped_df = grouped_df[['Unsigned error', 'Head distance', 'Signed error', 'Head rotation', 'Age']]
        plt.figure(figsize = (28, 28))
        #if condition in conditions_CI:
        #    plt.figure(figsize = (28, 28))
        #else:
        #    plt.figure(figsize = (42, 28))
        #sns.set(font_scale=5)
        #plt.tight_layout()
        correlation_matrix, p_value_matrix = spearmanr(grouped_df)
        correlation_df = pd.DataFrame(correlation_matrix[:4, :], index=grouped_df.columns[:4], 
                                    columns=grouped_df.columns)
        p_value_df = pd.DataFrame(p_value_matrix[:4, :], index=grouped_df.columns[:4], columns=grouped_df.columns)
        bright_colormap = LinearSegmentedColormap.from_list("bright_rd_bu_r", ["#ffcccc", "#ff6666", "#cc0000", "#0000cc", "#6666ff", "#ccccff"])
        g = sns.heatmap(correlation_df, cmap="RdBu_r", linewidths=.5, fmt=".2f", annot=False, annot_kws={"size": 20}, 
                        cbar=False)
        g.set_title(titles[nn], fontsize=100)
        for i in range(len(correlation_df)):
            for j in range(len(correlation_df.columns)):
                corr_value = correlation_df.iloc[i, j]
                p_value = p_value_df.iloc[i, j]
                if corr_value > -0.1:
                    color = 'black'
                else:
                    color = 'white'
                
                if p_value < 0.05:
                    annotation_text = f"{corr_value:.2f}* \n p = {p_value:.2f}"
                else:
                    annotation_text = f"{corr_value:.2f} \n p = {p_value:.2f}"
                if i == j:
                    annotation_text = ""
                g.text(j + 0.5, i + 0.5, annotation_text, ha='center', va='center', color=color, fontsize=80)
        sns.set_theme(style='white')
        #g = sns.heatmap(grouped_df.corr(method='spearman'), cmap = "RdBu_r", linewidths = .5, fmt=".2f", annot = True, annot_kws={"size": 20})
        g.tick_params(axis='both', labelsize=80)
        g.set_xticklabels(['UE', 'HD', 'SE', 'HR', 'Age'], 
                          rotation=0, ha='right')
        g.set_yticklabels(['UE', 'HD', 'SE', 'HR'], rotation=0, ha='right')
        plt.savefig('heatmap_' + condition + '.png')
        plt.close()
    mpl.rcParams['axes.prop_cycle'] = mpl.rcParamsDefault['axes.prop_cycle']
    mpl.rcParams['figure.dpi'] = 300

if want_stats:
    # CSVS FOR STATS 
    text_median_name = 'STATS_MEDIANS_boxplot.txt'
    text_median_path = text_median_name
    text_median_anova = open(text_median_path, 'w')
    anova_median_txt = []

    text_file_angles_name = 'STATS_ANGLES_boxplot.txt'
    text_file_angles_path = text_file_angles_name
    text_file_angles = open(text_file_angles_path, 'w')
    anova_data_angles_txt = []

    text_file_diff_angles_name = 'STATS_DIFFERENT_ANGLE_boxplot.txt'
    text_file_diff_angles_path = text_file_diff_angles_name
    text_file_diff_angles = open(text_file_diff_angles_path, 'w')
    anova_data_diff_angles_txt = []


# BOXPLOTS AND STATISTICS FOR EACH CONDITION WRT TARGETS
for n, var in enumerate(variables):
    data_for_all_targets = []
    data_for_all_targets_CI_HA = []
    for ncond, condition in enumerate(conditions):
    #for condition in conditions:
        filtered_data_condition = dataset_medians[(dataset_medians['Condition'] == condition) & 
                                                  (dataset_medians['Variable'] == var)]
        if want_qq:
            residual = filtered_data_condition['Median'] - np.mean(filtered_data_condition['Median'])
            qqplot = sm.qqplot(residual, line='q')
            name_qq = 'Q-Q ' + var_names[n] + ' ' + titles[ncond]
            plt.title(name_qq)
            plt.savefig(name_qq + '.png')
            plt.close()
        if want_stats:
            group_data = [
                filtered_data_condition[
                    filtered_data_condition['Target'] == group][
                        'Median'].values for group in filtered_data_condition['Target'].unique()]
            k = len(group_data)
            nn = np.sum([len(group) for group in group_data])
            df_between = k - 1
            df_within = nn - k
            bonfi = 0.05 / math.comb(k, 2)
            anova_data_diff_angles_txt.append(
                'STATS FOR VARIABLE ' + var + ' in condition '+ condition + ' between Targets')
            anova_data_diff_angles_txt.append('\n')
            anova_data_diff_angles_txt.append('BONFERRONI P-CORRECTED VALUE ' + str(bonfi))
            anova_data_diff_angles_txt.append('\n')
            anova_data_diff_angles_txt.append('CRONBACH ALPHA FOR VARIABLE ' + var)
            anova_data_diff_angles_txt.append('\n')
            cronbach = pg.cronbach_alpha(data=filtered_data_condition[[
                'Median', 'Participant', 'Age', 'Experience_DX', 'Experience_SX', 'Threshold_wo_500_DX', 
                'Threshold_wo_1000_DX', 'Threshold_wo_2000_DX', 'Threshold_wo_4000_DX', 'Threshold_wo_500_SX', 
                'Threshold_wo_1000_SX', 'Threshold_wo_2000_SX', 'Threshold_wo_4000_SX', 'Threshold_w_500_DX', 
                'Threshold_w_1000_DX', 'Threshold_w_2000_DX', 'Threshold_w_4000_DX', 'Threshold_w_500_SX', 
                'Threshold_w_1000_SX', 'Threshold_w_2000_SX', 'Threshold_w_4000_SX']], scores='Median', 
                subject='Participant', ci=0.95)
            anova_data_diff_angles_txt.append(convertTuple(cronbach))
            anova_data_diff_angles_txt.append('\n')
            anova_data_diff_angles_txt.append('NORMALITY CHECK FOR VARIABLE ' + var + ' BETWEEN Targets')
            anova_data_diff_angles_txt.append('\n')
            normi_data = filtered_data_condition.copy()
            for targi in filtered_data_condition['Target'].unique():
                residual = filtered_data_condition[
                    filtered_data_condition['Target'] == targi]['Median'] - np.mean(
                        filtered_data_condition[filtered_data_condition['Target'] == targi]['Median'])
                #normi_data[normi_data['Target'] == targi]['Median'] = residual
                normi_data.loc[normi_data['Target'] == targi, 'Median'] = residual
            normi = pg.normality(data=normi_data, dv='Median', group='Target')
            anova_data_diff_angles_txt.append(normi.to_string())
            anova_data_diff_angles_txt.append('\n')
            true_count = list(normi['normal']).count(True)
            false_count = list(normi['normal']).count(False)
            if false_count > true_count:
            #if False in list(normi['normal']):
                anova_data_diff_angles_txt.append('FRIEDMAN TEST FOR VARIABLE ' + var + ' BETWEEN Targets')
                anova_data_diff_angles_txt.append('\n')
                #fried = pg.friedman(data=filtered_data_condition, dv='Median', subject='Participant', within='Target')
                intersection_parts = filtered_data_condition[
                    filtered_data_condition['Target'] == targets[0]]['Participant']
                for tar in targets:
                    parties = filtered_data_condition[filtered_data_condition['Target'] == tar]['Participant']
                    intersection_parts = list(set(intersection_parts).intersection(parties))
                data = []
                sample_size = 0
                for tar in targets:
                    data_target = filtered_data_condition[filtered_data_condition['Target'] == tar]
                    data_to_append = np.asarray(
                        data_target[data_target['Participant'].isin(intersection_parts)]['Median'])
                    sample_size += len(data_to_append)
                    data.append(data_to_append)
                fried = friedmanchisquare(*data)
                result_str = "F = {}, pval = {})".format(fried.statistic, fried.pvalue)
                anova_data_diff_angles_txt.append('NUMEROSITY: ' + str(len(intersection_parts)))
                anova_data_diff_angles_txt.append('\n')
                anova_data_diff_angles_txt.append(result_str)
                anova_data_diff_angles_txt.append('\n')
                w = fried.statistic / (sample_size * (len(data[-1])-1))
                anova_data_diff_angles_txt.append('EFFECT SIZE: ' + str(w))
                anova_data_diff_angles_txt.append('\n')
                pval = fried.pvalue
                if pval < 0.05:
                    anova_data_diff_angles_txt.append('NEMENYI TEST FOR VARIABLE ' + var + ' BETWEEN Targets')
                    anova_data_diff_angles_txt.append('\n')
                    data = np.asarray(data)
                    pairwise = sp.posthoc_nemenyi_friedman(data.T)
                    anova_data_diff_angles_txt.append(pairwise.to_string())
                    anova_data_diff_angles_txt.append('\n')
                    meanRanks, pValues = test_nemenyi(data)
                    anova_data_diff_angles_txt.append('Nemenyi mean ranks ')
                    anova_data_diff_angles_txt.append(np.array2string(meanRanks, formatter={'float_kind':lambda x: "%.2f" % x}))
                    anova_data_diff_angles_txt.append('p-values ')
                    anova_data_diff_angles_txt.append(np.array2string(pValues, formatter={'float_kind':lambda x: "%.4f" % x}))
                    anova_data_diff_angles_txt.append('\n')
                    efx = pg.compute_effsize(data, paired=True, eftype='hedges')
                    anova_data_diff_angles_txt.append('EFFECT SIZE')
                    anova_data_diff_angles_txt.append(str(efx))
                    anova_data_diff_angles_txt.append('\n')
                    
            else:
                anova_data_diff_angles_txt.append('SPHERICITY CHECK FOR VARIABLE ' + var + ' CONDITION ' + condition)
                spher = pg.sphericity(filtered_data_condition, dv='Median', subject='Participant', within=['Target'])
                result_str = "Sphericity = {}, W = {}, chi2 = {}, dof = {}, pval = {})".format(spher.spher, spher.W, 
                                                                                                spher.chi2, spher.dof, 
                                                                                                spher.pval)
                anova_data_diff_angles_txt.append('\n')
                anova_data_diff_angles_txt.append(result_str)
                anova_data_diff_angles_txt.append('\n')
                anova_data_diff_angles_txt.append(
                    'HOMOSCEDASTICITY CHECK FOR VARIABLE ' + var + ' CONDITION ' + condition)
                anova_data_diff_angles_txt.append('\n')
                homo = pg.homoscedasticity(data=filtered_data_condition, dv='Median', group='Target')
                anova_data_diff_angles_txt.append(homo.to_string())
                anova_data_diff_angles_txt.append('\n')
                anova_data_diff_angles_txt.append('LEVENE DEGREES OF FREEDOM BETWEEN ' + str(df_between))
                anova_data_diff_angles_txt.append('\n')
                anova_data_diff_angles_txt.append('LEVENE DEGREES OF FREEDOM WITHIN ' + str(df_within))
                anova_data_diff_angles_txt.append('\n')
                if spher.spher:
                    anova_data_diff_angles_txt.append('RM ANOVA FOR VARIABLE ' + var)
                    anova_data_diff_angles_txt.append('\n')
                    aov = pg.rm_anova(data=filtered_data_condition, dv='Median', within='Target', 
                                        subject='Participant', detailed=True, effsize='np2')
                    anova_data_diff_angles_txt.append(aov.to_string())
                    anova_data_diff_angles_txt.append('\n')
                    alpha = 0.05
                    effect_size = aov['np2']
                    anova_data_diff_angles_txt.append('EFFECT SIZE ' + str(effect_size)) 
                    anova_data_diff_angles_txt.append('\n')
                    nobs = len(filtered_data_condition)
                    power_analysis = FTestAnovaPower().solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, 
                                                                    k_groups=len(
                                                                        filtered_data_condition['Target'].unique()))
                    anova_data_diff_angles_txt.append('ANOVA POWER ' + str(power_analysis[0])) 
                    anova_data_diff_angles_txt.append('\n')
                    if aov['p-unc'][0] < 0.05:
                        anova_data_diff_angles_txt.append('PAIRWISE TEST FOR VARIABLE ' + var + ' BETWEEN Targets')
                        anova_data_diff_angles_txt.append('\n')
                        pairwise = pg.pairwise_tests(data=filtered_data_condition, dv='Median', between='Target', 
                                                    subject='Participant', parametric=True, marginal=True, 
                                                    alpha=0.05, alternative='two-sided', padjust='bonf', 
                                                    effsize='hedges', correction='auto', return_desc=True, 
                                                    interaction=True, within_first=True)
                        anova_data_diff_angles_txt.append(pairwise.to_string())
                        anova_data_diff_angles_txt.append('\n')
                else:
                    anova_data_diff_angles_txt.append('GREENHOUSE RM ANOVA FOR VARIABLE ' + var)
                    anova_data_diff_angles_txt.append('\n')
                    aov = pg.rm_anova(data=filtered_data_condition, dv='Median', within='Target', 
                                        subject='Participant', detailed=True, effsize='np2',
                                        correction=True)
                    anova_data_diff_angles_txt.append(aov.to_string())
                    anova_data_diff_angles_txt.append('\n')
                    alpha = 0.05
                    effect_size = aov['np2']
                    anova_data_diff_angles_txt.append('EFFECT SIZE ' + str(effect_size)) 
                    anova_data_diff_angles_txt.append('\n')
                    nobs = len(filtered_data_condition)
                    power_analysis = FTestAnovaPower().solve_power(effect_size=effect_size, nobs=nobs, alpha=alpha, 
                                                                    k_groups=len(
                                                                        filtered_data_condition['Target'].unique()))
                    anova_data_diff_angles_txt.append('ANOVA POWER ' + str(power_analysis[0])) 
                    anova_data_diff_angles_txt.append('\n')
                    if aov['p-GG-corr'][0] < alpha:
                        anova_data_diff_angles_txt.append('PAIRWISE TEST FOR VARIABLE ' + var + ' BETWEEN Targets')
                        anova_data_diff_angles_txt.append('\n')
                        pairwise = pg.pairwise_tests(data=filtered_data_condition, dv='Median', between='Target', 
                                                    subject='Participant', parametric=True, marginal=True, 
                                                    alpha=0.05, alternative='two-sided', padjust='bonf', 
                                                    effsize='hedges', correction='auto', 
                                                    return_desc=True, interaction=True, within_first=True)
                        anova_data_diff_angles_txt.append(pairwise.to_string())
                        anova_data_diff_angles_txt.append('\n')

    # BOXPLOTS AND STATISTICS FOR EACH ANGLE BETWEEN DIFFERENT CONDITIONS
    for target in targets:
        filtered_data = dataset_medians[(dataset_medians['Target'] == target) & (dataset_medians['Variable'] == var)]
        if want_stats:
            #COMMENT OR UNCOMMENT FOR THE THREE COMPARISONS
            #filtered_data = filtered_data[filtered_data['Condition'].isin(conditions_HA)]
            #filtered_data = filtered_data[filtered_data['Condition'].isin(conditions_ON)]
            filtered_data = filtered_data[filtered_data['Condition'].isin(conditions_CI)]
            group_data = [filtered_data[filtered_data['Condition'] == group
                                        ]['Median'].values for group in filtered_data['Condition'].unique()]
            k = len(group_data)
            nn = np.sum([len(group) for group in group_data])
            df_between = k - 1
            df_within = nn - k
            bonfi = 0.05 / math.comb(k, 2)
            anova_data_angles_txt.append('STATS FOR VARIABLE ' + var + ' for target ' + str(target) + ' between Conditions')
            anova_data_angles_txt.append('\n')
            anova_data_angles_txt.append('BONFERRONI P-CORRECTED VALUE ' + str(bonfi))
            anova_data_angles_txt.append('\n')
            anova_data_angles_txt.append('CRONBACH ALPHA FOR VARIABLE')
            anova_data_angles_txt.append('\n')
            cronbach = pg.cronbach_alpha(data=filtered_data[[
                'Median', 'Participant', 'Age', 'Experience_DX', 'Experience_SX', 'Threshold_wo_500_DX', 
                'Threshold_wo_1000_DX', 'Threshold_wo_2000_DX', 'Threshold_wo_4000_DX', 'Threshold_wo_500_SX', 
                'Threshold_wo_1000_SX', 'Threshold_wo_2000_SX', 'Threshold_wo_4000_SX', 'Threshold_w_500_DX', 
                'Threshold_w_1000_DX', 'Threshold_w_2000_DX', 'Threshold_w_4000_DX', 'Threshold_w_500_SX', 
                'Threshold_w_1000_SX', 'Threshold_w_2000_SX', 'Threshold_w_4000_SX']], 
                scores='Median', subject='Participant', ci=0.95)
            anova_data_angles_txt.append(convertTuple(cronbach))
            anova_data_angles_txt.append('\n')
            conds = filtered_data['Condition'].unique()
            if len(conds) > 2:
                anova_data_angles_txt.append('NORMALITY CHECK FOR VARIABLE ' + var + ' BETWEEN Conditions')
                anova_data_angles_txt.append('\n')
                normi_data = filtered_data.copy()
                normi2_list = []
                for condi in conds:
                    residual = filtered_data[
                        filtered_data['Condition'] == condi]['Median'] - np.mean(
                            filtered_data[filtered_data['Condition'] == condi]['Median'])
                    #normi_data[normi_data['Condition'] == condi]['Median'] = residual
                    normi_data.loc[normi_data['Condition'] == condi, 'Median'] = residual
                    normi2 = anderson(x=residual, dist='norm')
                    normi2_list.append(normi2.statistic - normi2.critical_values[2] < 0)
                    result_str = "anderson = {}, critical = {}".format(
                        normi2.statistic, normi2.critical_values[2])
                    anova_data_angles_txt.append(result_str)
                    anova_data_angles_txt.append('\n')
                normi = pg.normality(data=normi_data, dv='Median', group='Condition')
                anova_data_angles_txt.append(normi.to_string())
                anova_data_angles_txt.append('\n')
                #if False in list(normi['normal']):
                if False in normi2_list:
                    anova_data_angles_txt.append('FRIEDMAN TEST FOR VARIABLE ' + var + ' BETWEEN Conditions')
                    anova_data_angles_txt.append('\n')
                    #fried = pg.friedman(data=filtered_data, dv='Median', subject='Participant', within='Condition')
                    #anova_data_angles_txt.append(fried.to_string())
                    #anova_data_angles_txt.append('\n')
                    intersection_parts = filtered_data[filtered_data['Condition'] == conds[0]]['Participant']
                    for con in conds:
                        parties = filtered_data[filtered_data['Condition'] == con]['Participant']
                        intersection_parts = list(set(intersection_parts).intersection(parties))
                    data = []
                    sample_size = 0
                    for con in conds:
                        data_cond = filtered_data[filtered_data['Condition'] == con]
                        data_to_append = np.asarray(
                            data_cond[data_cond['Participant'].isin(intersection_parts)]['Median'])
                        sample_size += len(data_to_append)
                        data.append(data_to_append)
                    fried = friedmanchisquare(*data)
                    result_str = "F = {}, pval = {})".format(fried.statistic, fried.pvalue)
                    anova_data_angles_txt.append('NUMEROSITY: ' + str(len(intersection_parts)))
                    anova_data_angles_txt.append('\n')
                    anova_data_angles_txt.append(result_str)
                    anova_data_angles_txt.append('\n')
                    w = fried.statistic / (sample_size * (len(data[-1])-1))
                    anova_data_angles_txt.append('EFFECT SIZE: ' + str(w))
                    anova_data_angles_txt.append('\n')
                    pval = fried.pvalue
                    if pval < 0.05:
                        anova_data_angles_txt.append('NEMENYI TEST FOR VARIABLE ' + var + ' BETWEEN Conditions')
                        anova_data_angles_txt.append('\n')
                        data = np.asarray(data)
                        pairwise = sp.posthoc_nemenyi_friedman(data.T)
                        anova_data_angles_txt.append(pairwise.to_string())
                        anova_data_angles_txt.append('\n')
                        meanRanks, pValues = test_nemenyi(data)
                        anova_data_angles_txt.append('Nemenyi mean ranks ')
                        anova_data_angles_txt.append(np.array2string(meanRanks, formatter={'float_kind':lambda x: "%.2f" % x}))
                        anova_data_angles_txt.append('p-values ')
                        anova_data_angles_txt.append(np.array2string(pValues, formatter={'float_kind':lambda x: "%.4f" % x}))
                        anova_data_angles_txt.append('\n')
                        efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[0]]['Median'], 
                                                 filtered_data[filtered_data['Condition'] == conds[1]]['Median'],
                                                 paired=True, eftype='hedges')
                        anova_data_angles_txt.append('EFFECT SIZE 0 1')
                        anova_data_angles_txt.append(str(efx))
                        anova_data_angles_txt.append('\n')
                        efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[0]]['Median'], 
                                                 filtered_data[filtered_data['Condition'] == conds[2]]['Median'],
                                                 paired=True, eftype='hedges')
                        anova_data_angles_txt.append('EFFECT SIZE 0 2')
                        anova_data_angles_txt.append(str(efx))
                        anova_data_angles_txt.append('\n')
                        #efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[0]]['Median'], 
                        #                         filtered_data[filtered_data['Condition'] == conds[3]]['Median'],
                        #                         paired=True, eftype='hedges')
                        #anova_data_angles_txt.append('EFFECT SIZE 0 3')
                        #anova_data_angles_txt.append(str(efx))
                        #anova_data_angles_txt.append('\n')
                        efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[1]]['Median'], 
                                                 filtered_data[filtered_data['Condition'] == conds[2]]['Median'],
                                                 paired=True, eftype='hedges')
                        anova_data_angles_txt.append('EFFECT SIZE 1 2')
                        anova_data_angles_txt.append(str(efx))
                        anova_data_angles_txt.append('\n')
                        #efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[1]]['Median'], 
                        #                         filtered_data[filtered_data['Condition'] == conds[3]]['Median'],
                        #                         paired=True, eftype='hedges')
                        #anova_data_angles_txt.append('EFFECT SIZE 1 3')
                        #anova_data_angles_txt.append(str(efx))
                        #anova_data_angles_txt.append('\n')
                        #efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[2]]['Median'], 
                        #                         filtered_data[filtered_data['Condition'] == conds[3]]['Median'],
                        #                         paired=True, eftype='hedges')
                        #anova_data_angles_txt.append('EFFECT SIZE 2 3')
                        #anova_data_angles_txt.append(str(efx))
                        #anova_data_angles_txt.append('\n')
                else:
                    sphericity = False
                    anova_data_angles_txt.append('SPHERICITY CHECK FOR VARIABLE ' + var + ' TARGET ' + str(target))
                    spher = pg.sphericity(filtered_data, dv='Median', subject='Participant', within='Condition')
                    result_str = "Sphericity = {}, W = {}, chi2 = {}, dof = {}, pval = {})".format(
                        spher[0], spher[1], spher[2], spher[3], spher[4])
                    anova_data_angles_txt.append('\n')
                    anova_data_angles_txt.append(result_str)
                    anova_data_angles_txt.append('\n')
                    sphericity = spher[0]
                    anova_data_angles_txt.append('HOMOSCEDASTICITY CHECK FOR VARIABLE ' + var + ' TARGET ' + str(target))
                    anova_data_angles_txt.append('\n')
                    homo = pg.homoscedasticity(data=filtered_data, dv='Median', group='Condition')
                    anova_data_angles_txt.append(homo.to_string())
                    anova_data_angles_txt.append('\n')
                    anova_data_angles_txt.append('LEVENE DEGREES OF FREEDOM BETWEEN ' + str(df_between))
                    anova_data_angles_txt.append('\n')
                    anova_data_angles_txt.append('LEVENE DEGREES OF FREEDOM WITHIN ' + str(df_within))
                    anova_data_angles_txt.append('\n')
                    if sphericity:
                        anova_data_angles_txt.append('RM ANOVA FOR VARIABLE ' + var)
                        anova_data_angles_txt.append('\n')
                        aov = pg.rm_anova(data=filtered_data, dv='Median', within='Condition', 
                                        subject='Participant', detailed=True, effsize='np2')
                        anova_data_angles_txt.append(aov.to_string())
                        anova_data_angles_txt.append('\n')
                        alpha = 0.05
                        effect_size = aov['np2']
                        anova_data_angles_txt.append('EFFECT SIZE ' + str(effect_size)) 
                        anova_data_angles_txt.append('\n')
                        nobs = len(filtered_data)
                        power_analysis = FTestAnovaPower().solve_power(effect_size=effect_size, nobs=nobs, 
                                                                    alpha=alpha, k_groups=len(
                                                                        filtered_data['Condition'].unique()))
                        anova_data_angles_txt.append('ANOVA POWER ' + str(power_analysis[0])) 
                        anova_data_angles_txt.append('\n')
                        if aov['p-unc'][0] < 0.05:
                            anova_data_angles_txt.append('PAIRWISE TEST FOR VARIABLE ' + var + ' BETWEEN Conditions')
                            anova_data_angles_txt.append('\n')
                            pairwise = pg.pairwise_tests(data=filtered_data, dv='Median', within='Condition', 
                                                        subject='Participant', parametric=True, marginal=True, 
                                                        alpha=0.05, alternative='two-sided', padjust='bonf', 
                                                        effsize='hedges', correction='auto', return_desc=True, 
                                                        interaction=True, within_first=True)
                            anova_data_angles_txt.append(pairwise.to_string())
                            anova_data_angles_txt.append('\n')
                    else:
                        anova_data_angles_txt.append('GREENHOUSE RM ANOVA FOR VARIABLE ' + var)
                        anova_data_angles_txt.append('\n')
                        aov = pg.rm_anova(data=filtered_data, dv='Median', within='Condition', 
                                        subject='Participant', correction=True, detailed=True, effsize='np2')
                        anova_data_angles_txt.append(aov.to_string())
                        anova_data_angles_txt.append('\n')
                        alpha = 0.05
                        effect_size = aov['np2']
                        anova_data_angles_txt.append('EFFECT SIZE ' + str(effect_size)) 
                        anova_data_angles_txt.append('\n')
                        nobs = len(filtered_data)
                        power_analysis = FTestAnovaPower().solve_power(effect_size=effect_size, nobs=nobs, 
                                                                        alpha=alpha, k_groups=len(
                                                                            filtered_data['Condition'].unique()))
                        anova_data_angles_txt.append('ANOVA POWER ' + str(power_analysis[0])) 
                        anova_data_angles_txt.append('\n')
                        if aov['p-GG-corr'][0] < alpha:
                            anova_data_angles_txt.append('PAIRWISE FOR VARIABLE ' + var + ' BETWEEN Conditions')
                            anova_data_angles_txt.append('\n')
                            pairwise = pg.pairwise_tests(data=filtered_data, dv='Median', 
                                                        between='Condition', 
                                                        subject='Participant', parametric=True, marginal=True, 
                                                        alpha=0.05, alternative='two-sided', padjust='bonf', 
                                                        effsize='hedges', correction='auto', 
                                                        return_desc=True, interaction=True, within_first=True)
                            anova_data_angles_txt.append(pairwise.to_string())
                            anova_data_angles_txt.append('\n')
            else:
                anova_data_angles_txt.append('NORMALITY CHECK FOR VARIABLE ' + var + ' BETWEEN Conditions')
                anova_data_angles_txt.append('\n')
                normi_data = filtered_data.copy()
                normi2_list = []
                for condi in filtered_data['Condition'].unique():
                    residual = filtered_data[
                        filtered_data['Condition'] == condi]['Median'] - np.mean(
                            filtered_data[filtered_data['Condition'] == condi]['Median'])
                    #normi_data[normi_data['Condition'] == condi]['Median'] = residual
                    normi_data.loc[normi_data['Condition'] == condi, 'Median'] = residual
                    normi2 = anderson(x=residual, dist='norm')
                    normi2_list.append(normi2.statistic - normi2.critical_values[2] < 0)
                    result_str = "anderson = {}, critical = {}".format(
                        normi2.statistic, normi2.critical_values[2])
                    anova_data_angles_txt.append(result_str)
                    anova_data_angles_txt.append('\n')
                normi = pg.normality(data=normi_data, dv='Median', group='Condition')
                anova_data_angles_txt.append(normi.to_string())
                anova_data_angles_txt.append('\n')
                anova_data_angles_txt.append('HOMOSCEDASTICITY CHECK FOR VARIABLE ' + var + ' TARGET ' + str(target))
                anova_data_angles_txt.append('\n')
                homo = pg.homoscedasticity(data=filtered_data, dv='Median', group='Condition')
                anova_data_angles_txt.append(homo.to_string())
                anova_data_angles_txt.append('\n')
                anova_data_angles_txt.append('LEVENE DEGREES OF FREEDOM BETWEEN ' + str(df_between))
                anova_data_angles_txt.append('\n')
                anova_data_angles_txt.append('LEVENE DEGREES OF FREEDOM WITHIN ' + str(df_within))
                anova_data_angles_txt.append('\n')
                #if False in list(normi['normal']):
                if False in normi2_list:
                    if list(homo['equal_var'])[0]:
                        anova_data_angles_txt.append('PAIRWISE FOR VARIABLE ' + var + ' BETWEEN Conditions')
                        anova_data_angles_txt.append('\n')
                        pairwise = pg.pairwise_tests(data=filtered_data, dv='Median', between='Condition', 
                                                    subject='Participant', parametric=False, marginal=True, 
                                                    alpha=0.05, alternative='two-sided', padjust='bonf', 
                                                    effsize='hedges', correction='auto', 
                                                    return_desc=True, interaction=True, within_first=True)
                        anova_data_angles_txt.append(pairwise.to_string())
                        anova_data_angles_txt.append('\n')
                    else:
                        anova_data_angles_txt.append('YUEN T-TEST FOR VARIABLE ' + var)
                        anova_data_angles_txt.append('\n')
                        aov = stats.ttest_ind(filtered_data[filtered_data['Condition'] == conds[0]]['Median'],
                                    filtered_data[filtered_data['Condition'] == conds[1]]['Median'], equal_var=False, 
                                    trim=0.1)
                        aov_str = "statistic = {}, pvalue = {})".format(aov[0], aov[1])
                        anova_data_angles_txt.append(aov_str)
                        anova_data_angles_txt.append('\n')
                        efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[0]]['Median'],
                                                 filtered_data[filtered_data['Condition'] == conds[1]]['Median'],
                                                 paired=False, eftype='hedges')
                        anova_data_angles_txt.append('EFFECT SIZE')
                        anova_data_angles_txt.append(str(efx))
                        anova_data_angles_txt.append('\n')
                else:
                    if list(homo['equal_var'])[0]:
                        anova_data_angles_txt.append('T-TEST FOR VARIABLE ' + var)
                        anova_data_angles_txt.append('\n')
                        aov = pg.ttest(filtered_data[filtered_data['Condition'] == conds[0]]['Median'],
                                    filtered_data[filtered_data['Condition'] == conds[1]]['Median'])
                        anova_data_angles_txt.append(aov.to_string())
                        anova_data_angles_txt.append('\n')
                        efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[0]]['Median'],
                                                 filtered_data[filtered_data['Condition'] == conds[1]]['Median'],
                                                 paired=False, eftype='hedges')
                        anova_data_angles_txt.append('EFFECT SIZE')
                        anova_data_angles_txt.append(str(efx))
                        anova_data_angles_txt.append('\n')
                    else:
                        anova_data_angles_txt.append('WELCH T-TEST FOR VARIABLE ' + var)
                        anova_data_angles_txt.append('\n')
                        aov = pg.ttest(filtered_data[filtered_data['Condition'] == conds[0]]['Median'],
                                       filtered_data[filtered_data['Condition'] == conds[1]]['Median'],
                                       correction=True)
                        anova_data_angles_txt.append(aov.to_string())
                        anova_data_angles_txt.append('\n')
                        efx = pg.compute_effsize(filtered_data[filtered_data['Condition'] == conds[0]]['Median'],
                                                 filtered_data[filtered_data['Condition'] == conds[1]]['Median'],
                                                 paired=False, eftype='hedges')
                        anova_data_angles_txt.append('EFFECT SIZE')
                        anova_data_angles_txt.append(str(efx))
                        anova_data_angles_txt.append('\n')    
        if want_figures:
            #filtered_data = dataset_medians[(dataset_medians['Target'] == target) & (dataset_medians['Variable'] == var)]
            for condition in conditions:
                filtered_data_var = filtered_data[(filtered_data['Condition'] == condition)]
                data_for_all_targets.append(np.exp(filtered_data_var['Median']) + min_values[n] - 1)
                if condition in ['ICSX_ICDX', 'PASX_PADX']:
                    data_for_all_targets_CI_HA.append(np.exp(filtered_data_var['Median']) + min_values[n] - 1)

    # BOXPLOTS AND STATISTICS FOR MEDIANS WRT ANGLES BETWEEN CONDITIONS
    groupy = []
    grouped = clean_dataset.groupby(['Participant', 'Condition'], as_index=False).agg(
        {'Signed_error': 'median', 'Unsigned_error': 'median', 'Head_rotation': 'median', 'Head_distance': 'median'})
    grouped_for_plot = grouped[grouped['Condition'].isin(conditions)]
    #COMMENT OR UNCOMMENT FOR THE THREE COMPARISONS
    #grouped = grouped[grouped['Condition'].isin(conditions_HA)]
    #grouped = grouped[grouped['Condition'].isin(conditions_ON)]
    grouped = grouped[grouped['Condition'].isin(conditions_CI)]
    for condition in conditions:
        groupp = grouped_for_plot[(grouped_for_plot['Condition'] == condition)]
        groupy.append(np.exp(groupp[var]) + min_values[n] - 1)
    if want_stats:
        conds = grouped['Condition'].unique()
        if len(conds) > 2:
            group_data = [grouped[grouped['Condition'] == group
                                        ][var].values for group in grouped['Condition'].unique()]
            k = len(group_data)
            nn = np.sum([len(group) for group in group_data])
            df_between = k - 1
            df_within = nn - k
            bonfi = 0.05 / math.comb(k, 2)
            anova_median_txt.append('STATS FOR VARIABLE ' + var + ' MEDIAN wrt targets BETWEEN Conditions')
            anova_median_txt.append('\n')
            anova_median_txt.append('BONFERRONI P-CORRECTED VALUE ' + str(bonfi))
            anova_median_txt.append('\n')
            anova_median_txt.append('CRONBACH ALPHA FOR VARIABLE')
            anova_median_txt.append('\n')
            cronbach = pg.cronbach_alpha(data=grouped[[var, 'Participant']], scores=var, subject='Participant', ci=0.95)
            anova_median_txt.append(convertTuple(cronbach))
            anova_median_txt.append('\n')
            anova_median_txt.append('NORMALITY CHECK FOR VARIABLE ' + var + ' BETWEEN Conditions')
            anova_median_txt.append('\n')
            normi_data = grouped.copy()
            normi2_list = []
            for condi in grouped['Condition'].unique():
                residual = grouped[
                    grouped['Condition'] == condi][var] - np.mean(
                        grouped[grouped['Condition'] == condi][var])
                #normi_data[normi_data['Condition'] == condi][var] = residual
                normi_data.loc[normi_data['Condition'] == condi, var] = residual
                normi2 = anderson(x=residual, dist='norm')
                result_str = "anderson = {}, critical = {}".format(
                    normi2.statistic, normi2.critical_values[2])
                normi2_list.append(normi2.statistic - normi2.critical_values[2] < 0)
                anova_median_txt.append(result_str)
                anova_median_txt.append('\n')
            normi = pg.normality(data=normi_data, dv=var, group='Condition')
            anova_median_txt.append(normi.to_string())
            anova_median_txt.append('\n')
            #if False in list(normi['normal']):
            if False in normi2_list:
                anova_median_txt.append('FRIEDMAN TEST FOR VARIABLE ' + var + ' BETWEEN Conditions')
                anova_median_txt.append('\n')
                #fried = pg.friedman(data=grouped, dv=var, subject='Participant', within='Condition')
                #anova_data_angles_txt.append(fried.to_string())
                #anova_data_angles_txt.append('\n')
                intersection_parts = grouped[grouped['Condition'] == conds[0]]['Participant']
                for con in conds:
                    parties = grouped[grouped['Condition'] == con]['Participant']
                    intersection_parts = list(set(intersection_parts).intersection(parties))
                data = []
                sample_size = 0
                for con in conds:
                    data_cond = grouped[grouped['Condition'] == con]
                    data_to_append = np.asarray(
                        data_cond[data_cond['Participant'].isin(intersection_parts)][var])
                    sample_size += len(data_to_append)
                    data.append(data_to_append)
                fried = friedmanchisquare(*data)
                result_str = "F = {}, pval = {})".format(fried.statistic, fried.pvalue)
                anova_median_txt.append('NUMEROSITY: ' + str(len(intersection_parts)))
                anova_median_txt.append('\n')
                anova_median_txt.append(result_str)
                anova_median_txt.append('\n')
                w = fried.statistic / (sample_size * (len(data[-1])-1))
                anova_median_txt.append('EFFECT SIZE: ' + str(w))
                anova_median_txt.append('\n')
                pval = fried.pvalue
                if pval < 0.05:
                    anova_median_txt.append('NEMENYI TEST FOR VARIABLE ' + var + ' BETWEEN Conditions')
                    anova_median_txt.append('\n')
                    data = np.asarray(data)
                    pairwise = sp.posthoc_nemenyi_friedman(data.T)
                    anova_median_txt.append(pairwise.to_string())
                    anova_median_txt.append('\n')
                    meanRanks, pValues = test_nemenyi(data)
                    anova_median_txt.append('Nemenyi mean ranks ')
                    anova_median_txt.append(np.array2string(meanRanks, formatter={'float_kind':lambda x: "%.2f" % x}))
                    anova_median_txt.append('p-values ')
                    anova_median_txt.append(np.array2string(pValues, formatter={'float_kind':lambda x: "%.4f" % x}))
                    anova_median_txt.append('\n')
                    efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[0]][var], 
                                             grouped[grouped['Condition'] == conds[1]][var],
                                             paired=True, eftype='hedges')
                    anova_median_txt.append('EFFECT SIZE 0 1')
                    anova_median_txt.append(str(efx))
                    anova_median_txt.append('\n')
                    efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[0]][var], 
                                             grouped[grouped['Condition'] == conds[2]][var],
                                             paired=True, eftype='hedges')
                    anova_median_txt.append('EFFECT SIZE 0 2')
                    anova_median_txt.append(str(efx))
                    anova_median_txt.append('\n')
                    #efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[0]][var], 
                    #                         grouped[grouped['Condition'] == conds[3]][var],
                    #                         paired=True, eftype='hedges')
                    #anova_median_txt.append('EFFECT SIZE 0 3')
                    #anova_median_txt.append(str(efx))
                    #anova_median_txt.append('\n')
                    efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[1]][var], 
                                             grouped[grouped['Condition'] == conds[2]][var],
                                             paired=True, eftype='hedges')
                    anova_median_txt.append('EFFECT SIZE 1 2')
                    anova_median_txt.append(str(efx))
                    anova_median_txt.append('\n')
                    #efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[1]][var], 
                    #                         grouped[grouped['Condition'] == conds[3]][var],
                    #                         paired=True, eftype='hedges')
                    #anova_median_txt.append('EFFECT SIZE 1 3')
                    #anova_median_txt.append(str(efx))
                    #anova_median_txt.append('\n')
                    #efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[2]][var], 
                    #                         grouped[grouped['Condition'] == conds[3]][var],
                    #                         paired=True, eftype='hedges')
                    #anova_median_txt.append('EFFECT SIZE 2 3')
                    #anova_median_txt.append(str(efx))
                    #anova_median_txt.append('\n')
            else:
                anova_median_txt.append('SPHERICITY CHECK FOR VARIABLE ' + var + ' BETWEEN Conditions')
                spher = pg.sphericity(grouped, dv=var, subject='Participant', within='Condition')
                result_str = "Sphericity = {}, W = {}, chi2 = {}, dof = {}, pval = {})".format(
                    spher[0], spher[1], spher[2], spher[3], spher[4])
                anova_median_txt.append('\n')
                anova_median_txt.append(result_str)
                anova_median_txt.append('\n')
                sphericity = spher[0]
                anova_median_txt.append('LEVENE DEGREES OF FREEDOM BETWEEN ' + str(df_between))
                anova_median_txt.append('\n')
                anova_median_txt.append('LEVENE DEGREES OF FREEDOM WITHIN ' + str(df_within))
                anova_median_txt.append('\n')
                if sphericity:
                    anova_median_txt.append('RM ANOVA FOR VARIABLE ' + var)
                    anova_median_txt.append('\n')
                    aov = pg.rm_anova(data=grouped, dv=var, within='Condition', 
                                    subject='Participant', detailed=True, effsize='np2')
                    anova_median_txt.append(aov.to_string())
                    anova_median_txt.append('\n')
                    alpha = 0.05
                    effect_size = aov['np2']
                    anova_median_txt.append('EFFECT SIZE ' + str(effect_size)) 
                    anova_median_txt.append('\n')
                    nobs = len(filtered_data)
                    power_analysis = FTestAnovaPower().solve_power(effect_size=effect_size, nobs=nobs, 
                                                                    alpha=alpha, k_groups=len(
                                                                        grouped['Condition'].unique()))
                    anova_median_txt.append('ANOVA POWER ' + str(power_analysis[0])) 
                    anova_median_txt.append('\n')
                    if aov['p-unc'][0] < 0.05:
                        anova_median_txt.append('PAIRWISE TEST FOR VARIABLE ' + var + ' BETWEEN Conditions')
                        anova_median_txt.append('\n')
                        pairwise = pg.pairwise_tests(data=grouped, dv=var, within='Condition', 
                                                    subject='Participant', parametric=True, marginal=True, 
                                                    alpha=0.05, alternative='two-sided', padjust='bonf', 
                                                    effsize='hedges', correction='auto', return_desc=True, 
                                                    interaction=True, within_first=True)
                        anova_median_txt.append(pairwise.to_string())
                        anova_median_txt.append('\n')
                else:
                    anova_median_txt.append('GREENHOUSE RM ANOVA FOR VARIABLE ' + var)
                    anova_median_txt.append('\n')
                    aov = pg.rm_anova(data=grouped, dv=var, within='Condition', 
                                    subject='Participant', correction=True, detailed=True, effsize='np2')
                    anova_median_txt.append(aov.to_string())
                    anova_median_txt.append('\n')
                    alpha = 0.05
                    effect_size = aov['np2']
                    anova_median_txt.append('EFFECT SIZE ' + str(effect_size)) 
                    anova_median_txt.append('\n')
                    nobs = len(grouped)
                    power_analysis = FTestAnovaPower().solve_power(effect_size=effect_size, nobs=nobs, 
                                                                    alpha=alpha, k_groups=len(
                                                                        grouped['Condition'].unique()))
                    anova_median_txt.append('ANOVA POWER ' + str(power_analysis[0])) 
                    anova_median_txt.append('\n')
                    if aov['p-GG-corr'][0] < alpha:
                        anova_median_txt.append('PAIRWISE FOR VARIABLE ' + var + ' BETWEEN Conditions')
                        anova_median_txt.append('\n')
                        pairwise = pg.pairwise_tests(data=grouped, dv=var, 
                                                    between='Condition', 
                                                    subject='Participant', parametric=True, marginal=True, 
                                                    alpha=0.05, alternative='two-sided', padjust='bonf', 
                                                    effsize='hedges', correction='auto', 
                                                    return_desc=True, interaction=True, within_first=True)
                        anova_median_txt.append(pairwise.to_string())
                        anova_median_txt.append('\n')
        else:
            anova_median_txt.append('\n')
            anova_median_txt.append('NORMALITY CHECK FOR VARIABLE ' + var + ' BETWEEN Conditions')
            anova_median_txt.append('\n')
            normi_data = grouped.copy()
            normi2_list = []
            for condi in grouped['Condition'].unique():
                residual = grouped[
                    grouped['Condition'] == condi][var] - np.mean(
                        grouped[grouped['Condition'] == condi][var])
                normi_data.loc[normi_data['Condition'] == condi, var] = residual
                normi2 = anderson(x=residual, dist='norm')
                normi2_list.append(normi2.statistic - normi2.critical_values[2] < 0)
                result_str = "anderson = {}, critical = {}".format(
                    normi2.statistic, normi2.critical_values[2])
                anova_median_txt.append(result_str)
                anova_median_txt.append('\n')
            normi = pg.normality(data=normi_data, dv=var, group='Condition')
            anova_median_txt.append(normi.to_string())
            anova_median_txt.append('\n')
            anova_median_txt.append('HOMOSCEDASTICITY CHECK FOR VARIABLE ' + var + ' BETWEEN Conditions')
            anova_median_txt.append('\n')
            homo = pg.homoscedasticity(data=grouped, dv=var, group='Condition')
            anova_median_txt.append(homo.to_string())
            anova_median_txt.append('\n')
            anova_median_txt.append('LEVENE DEGREES OF FREEDOM BETWEEN ' + str(df_between))
            anova_median_txt.append('\n')
            anova_median_txt.append('LEVENE DEGREES OF FREEDOM WITHIN ' + str(df_within))
            anova_median_txt.append('\n')
            #if False in list(normi['normal']):
            if False in normi2_list:
                if list(homo['equal_var'])[0]:
                    anova_median_txt.append('PAIRWISE FOR VARIABLE ' + var + ' BETWEEN Conditions')
                    anova_median_txt.append('\n')
                    pairwise = pg.pairwise_tests(data=grouped, dv=var, between='Condition', 
                                                subject='Participant', parametric=False, marginal=True, 
                                                alpha=0.05, alternative='two-sided', padjust='bonf', 
                                                effsize='hedges', correction='auto', 
                                                return_desc=True, interaction=True, within_first=True)
                    anova_median_txt.append(pairwise.to_string())
                    anova_median_txt.append('\n')
                else:
                    anova_median_txt.append('YUEN T-TEST FOR VARIABLE ' + var)
                    anova_median_txt.append('\n')
                    aov = stats.ttest_ind(grouped[grouped['Condition'] == conds[0]][var],
                                            grouped[grouped['Condition'] == conds[1]][var], 
                                            equal_var=False, trim=0.1)
                    aov_str = "statistic = {}, pvalue = {})".format(aov[0], aov[1])
                    anova_median_txt.append(aov_str)
                    anova_median_txt.append('\n')
                    efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[0]][var],
                                             grouped[grouped['Condition'] == conds[1]][var],
                                             paired=False, eftype='hedges')
                    anova_median_txt.append('EFFECT SIZE')
                    anova_median_txt.append(str(efx))
                    anova_median_txt.append('\n')
            else:
                if list(homo['equal_var'])[0]:
                    anova_median_txt.append('T-TEST FOR VARIABLE ' + var)
                    anova_median_txt.append('\n')
                    aov = pg.ttest(grouped[grouped['Condition'] == conds[0]][var],
                                    grouped[grouped['Condition'] == conds[1]][var])
                    anova_median_txt.append(aov.to_string())
                    anova_median_txt.append('\n')
                    efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[0]][var],
                                             grouped[grouped['Condition'] == conds[1]][var],
                                             paired=False, eftype='hedges')
                    anova_median_txt.append('EFFECT SIZE')
                    anova_median_txt.append(str(efx))
                    anova_median_txt.append('\n')
                else:
                    anova_median_txt.append('WELCH T-TEST FOR VARIABLE ' + var)
                    anova_median_txt.append('\n')
                    aov = pg.ttest(grouped[grouped['Condition'] == conds[0]][var],
                                    grouped[grouped['Condition'] == conds[1]][var],
                                    correction=True)
                    anova_median_txt.append(aov.to_string())
                    anova_median_txt.append('\n')
                    efx = pg.compute_effsize(grouped[grouped['Condition'] == conds[0]][var],
                                             grouped[grouped['Condition'] == conds[1]][var],
                                             paired=False, eftype='hedges')
                    anova_median_txt.append('EFFECT SIZE')
                    anova_median_txt.append(str(efx))
                    anova_median_txt.append('\n')
    if want_figures:
        fig, ax = plt.subplots(figsize=(20, 10))
        bp = ax.boxplot(groupy, patch_artist=True)
        for box, color, pattern in zip(bp['boxes'], colors_plot, patterns):
            box.set(facecolor=color, linewidth=1.5, hatch=pattern)
        for median in bp['medians']:
            median.set_color('yellow')
            median.set_linewidth(4)
        ax.set_ylabel(y_labels[n], fontsize=40)
        legend_labels = ['CI L', 'CI ON', 'CI R', 'HA L', 'HA ON', 'NO HA', 'HA R']
        if var == 'Head_distance':
            ax.legend(bp['boxes'][:len(conditions)], legend_labels, loc='upper center', 
                    bbox_to_anchor=(0.75, 0.9), ncol=3, fontsize=30)
        elif var == 'Head_rotation':
            ax.legend(bp['boxes'][:len(conditions)], legend_labels, loc='upper center', 
                    bbox_to_anchor=(0.7, 0.16), ncol=3, fontsize=30)
        elif var == 'Signed_error':
            ax.legend(bp['boxes'][:len(conditions)], legend_labels, loc='best', ncol=3, fontsize=30)
        else:
            ax.legend(bp['boxes'][:len(conditions)], legend_labels, loc='best', ncol=3, fontsize=30)
        for medi in bp['medians']:
            x1plus, y1plus = medi.get_xydata()[1]
            if var == 'Head_distance':
                text = '{:.2f}'.format(y1plus)
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticks().round(1), fontsize=30)
            elif var == 'Head_rotation':
                text = '{:.1f}'.format(y1plus)
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
            elif var == 'Unsigned_error':
                text = '{:.1f}'.format(y1plus)
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
                labels_y = ax.get_yticklabels()
                labels_y[0] = ""
                ax.set_yticklabels(labels_y)
            else:
                text = '{:.1f}'.format(y1plus)
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
            ax.annotate(text, xy=(x1plus + 0.1, y1plus), fontsize=30)
        box_x_positions = [(item.get_xydata()[0][0] + item.get_xydata()[1][0]) * 0.5 for item in bp['medians']]
        whisker_values = [np.max(item.get_ydata()) for item in bp['whiskers']]
        whisker_values = whisker_values[1::2]
        fliers_values = [item.get_ydata() for item in bp['fliers']]
        fliers = []
        for nn, flier in enumerate(fliers_values):
            if flier.size > 0:
                fliers.append(np.max(flier))
            else:
                fliers.append(whisker_values[nn])
        ax.set_xticks([])
        plt.tight_layout()
        plot_filename = f'Boxplots_median_{var}.png'
        plt.savefig(plot_filename)
        plt.close()



        # BOXPLOTS PER EACH ANGLE OF CI AND HA CONDITIONS
        fig, ax = plt.subplots(figsize=(20, 10))
        targets_positions = []
        for ta in np.arange(5, 126, 10):
            targets_positions.append([ta - 1, ta + 1])
        targets_positions = [item for sublist in targets_positions for item in sublist]
        bp = ax.boxplot(data_for_all_targets_CI_HA, patch_artist=True, positions=targets_positions)
        colors_plot_ON = colors_ON * len(targets)
        for box, color in zip(bp['boxes'], colors_plot_ON):
            box.set(facecolor=color, linewidth=1.5)
        ax.set_ylabel(y_labels[n], fontsize=40)
        ax.set_xticks(range(5, len(targets) * (len(colors_all) + 3) - 4, len(colors_all) + 3))
        ax.set_xticklabels([str(int(target)) for target in targets], fontsize=30)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
        legend_labels = ['CI', 'HA']
        if var == 'Head_distance':
            ax.legend(bp['boxes'][:len(conditions)], legend_labels, loc='best', bbox_to_anchor=(0.75, 0.9), 
                    ncol=3, fontsize=30)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticks().astype(float).round(1))
            plt.setp(ax.get_yticklabels()[0], visible=False)
        elif var == 'Head_rotation':
            ax.legend(bp['boxes'][:len(conditions)], legend_labels, loc='best', bbox_to_anchor=(0.7, 0.16), 
                    ncol=3, fontsize=30)
        elif var == 'Unsigned_error':
            plt.setp(ax.get_yticklabels()[0], visible=False)
            ax.legend(bp['boxes'][:len(conditions)], legend_labels, loc='best', ncol=3, fontsize=30)
        else:
            ax.legend(bp['boxes'][:len(conditions)], legend_labels, loc='best', ncol=3, fontsize=30)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.xlabel('Target [°]', fontsize=30)
        plot_filename = f'Boxplots_angles_{var}.png'
        plt.savefig(plot_filename)
        plt.close()

if want_stats:
    # CLOSE THE STATS CSVS
    text_file_angles.writelines(anova_data_angles_txt)
    text_file_angles.close()
    text_file_diff_angles.writelines(anova_data_diff_angles_txt)
    text_file_diff_angles.close()
    text_median_anova.writelines(anova_median_txt)
    text_median_anova.close()

if want_figures:
    # BOXPLOTS FOR THE CONDITIONS OF THE TWO POPULATIONS
    conditions_PA = ['PASX_NOPADX', 'PASX_PADX', 'NOPASX_NOPADX', 'NOPASX_PADX']  
    conditions_IC = ['ICSX_NOICDX', 'ICSX_ICDX', 'NOICSX_ICDX']
    colors_PA = ['c', 'g', 'grey', 'm', 'w'] * len(targets)
    colors_IC = ['b', 'y', 'r', 'w'] * len(targets)
    for n, var in enumerate(variables):
        data_for_all_targets_CI = []
        data_for_all_targets_PA = []
        data_for_all_targets_CI_not_transf = []
        data_for_all_targets_PA_not_transf = []
        for target in targets:
            filtered_data = dataset_medians[(dataset_medians['Target'] == target) & (dataset_medians['Variable'] == var)]
            for condition in conditions_IC:
                filtered_data_var = filtered_data[(filtered_data['Condition'] == condition)]
                data_for_all_targets_CI.append(filtered_data_var['Median'])
                data_for_all_targets_CI_not_transf.append(np.exp(filtered_data_var['Median']) + min_values[n] - 1)
            data_for_all_targets_CI.append([])
            data_for_all_targets_CI_not_transf.append([])
            for condition in conditions_PA:
                filtered_data_var = filtered_data[(filtered_data['Condition'] == condition)]
                data_for_all_targets_PA.append(filtered_data_var['Median'])
                data_for_all_targets_PA_not_transf.append(np.exp(filtered_data_var['Median']) + min_values[n] - 1)
            data_for_all_targets_PA.append([])
            data_for_all_targets_PA_not_transf.append([])
        fig, ax = plt.subplots(figsize=(20, 10))
        bp = ax.boxplot(data_for_all_targets_CI_not_transf, patch_artist=True)
        for box, color in zip(bp['boxes'], colors_IC):
            box.set(facecolor=color, linewidth=1.5)
        ax.set_xticks(range(2, len(targets) * (len(conditions_IC) + 1) + 1, len(conditions_IC) + 1))
        ax.set_xticklabels([str(int(target))+'°' for target in targets], fontsize=30)
        ax.set_ylabel(y_labels[n], fontsize=40)
        if var == 'Head_distance':
                text = '{:.2f}'.format(y1plus)
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticks().round(1), fontsize=30)
                plt.setp(ax.get_yticklabels()[0], visible=False)
        elif var == 'Head_rotation':
                text = '{:.2f}'.format(y1plus)
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
        elif var == 'Unsigned_error':
                text = '{:.2f}'.format(y1plus)
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
                plt.setp(ax.get_yticklabels()[0], visible=False)
        else:
            text = '{:.0f}'.format(y1plus)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
            #if var == 'Unsigned_error':
            #    ax.set_yticks(range(0, 251))
            #    #ax.set_yticklabels(ax.get_yticks().astype(int))
        legend_labels = ['CI L', 'CI ON', 'CI R']
        ax.legend(bp['boxes'][:len(conditions_IC)], legend_labels, loc='best', fontsize=30)
        plt.tight_layout()
        plot_filename = f'Boxplots_CI_Variable_{var}.png'
        plt.savefig(plot_filename)
        plt.close()
        fig, ax = plt.subplots(figsize=(20, 10))
        bp = ax.boxplot(data_for_all_targets_PA_not_transf, patch_artist=True)
        for box, color in zip(bp['boxes'], colors_PA):
            box.set(facecolor=color, linewidth=1.5)
        ax.set_xticks(range(2, len(targets) * (len(conditions_PA) + 1) + 1, len(conditions_PA) + 1))
        ax.set_xticklabels([str(int(target))+'°' for target in targets], fontsize=30)
        ax.set_ylabel(y_labels[n], fontsize=40)
        if var == 'Head_distance':
            text = '{:.2f}'.format(y1plus)
            #ax.set_yticks(range(0, 2))
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticks().round(1), fontsize=30)
            plt.setp(ax.get_yticklabels()[0], visible=False)
        elif var == 'Head_rotation':
            text = '{:.2f}'.format(y1plus)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
        elif var == 'Unsigned_error':
            text = '{:.2f}'.format(y1plus)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
            plt.setp(ax.get_yticklabels()[0], visible=False)
        else:
            text = '{:.0f}'.format(y1plus)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=30)
        legend_labels = ['HA L', 'HA ON', 'NO HA', 'HA R']
        ax.legend(bp['boxes'][:len(conditions_PA)], legend_labels, loc='best', fontsize=30)
        plt.tight_layout()
        plot_filename = f'Boxplots_PA_Variable_{var}.png'
        plt.savefig(plot_filename)
        plt.close()


    # POLAR PLOTS FOR THE THREE DIFFERENT GROUPS OF CONDITIONS
    targets_strings = ['-90°', '-75°', '-60°', '-45°', '-30°', '-15°', '0°', '15°', '30°', '45°', '60°', '75°', '90°']
    conditions_PA = ['PASX_NOPADX', 'PASX_PADX', 'NOPASX_NOPADX', 'NOPASX_PADX']  
    conditions_IC = ['ICSX_NOICDX', 'ICSX_ICDX', 'NOICSX_ICDX']
    colors_PA = ['c', 'g', 'grey', 'm']
    colors_IC = ['b', 'y', 'r']
    legends_PA = ['HA L', 'HA ON', 'NO HA', 'HA R']
    legends_IC = ['CI L', 'CI ON', 'CI R']
    unit_measure = [' °', ' °', ' °', ' m']
    for n, var in enumerate(variables):
        data_for_all_targets_CI = []
        data_for_all_targets_CI_min = []
        data_for_all_targets_CI_max = []
        data_for_all_targets_PA_min = []
        data_for_all_targets_PA_max = []
        data_for_all_targets_PA = []
        angles_CI = []
        angles_PA = []
        angles = []
        cols_CI = []
        cols_PA = []
        legs_CI = []
        legs_PA = []
        filtered_data_all_targets = dataset_medians[(dataset_medians['Variable'] == var)]
        for target in targets:
            angles.append(-target * np.pi / 180 + np.pi * 0.5)
            filtered_data = dataset_medians[(dataset_medians['Target'] == target) & (dataset_medians['Variable'] == var)]
            for m, condition in enumerate(conditions_IC):
                filtered_data_var = filtered_data[(filtered_data['Condition'] == condition)]
                filtered_data_condition = filtered_data_all_targets[(
                    filtered_data_all_targets['Condition'] == condition)]
                data_for_all_targets_CI.append(np.median(filtered_data_var['Median']))
                data_for_all_targets_CI_min.append(np.min(filtered_data_var['Median']))
                data_for_all_targets_CI_max.append(np.max(filtered_data_var['Median']))
                angles_CI.append(-target * np.pi / 180 + np.pi * 0.5)
                cols_CI.append(colors_IC[m])
                #legs_CI.append(legends_IC[m] + ' ' + str(np.round(np.ptp(np.exp(
                #    filtered_data_condition['Median']) + min_values[n] - 1), 2)) + unit_measure[n])
                legs_CI.append(legends_IC[m])
            for m, condition in enumerate(conditions_PA):
                filtered_data_var = filtered_data[(filtered_data['Condition'] == condition)]
                filtered_data_condition = filtered_data_all_targets[(
                    filtered_data_all_targets['Condition'] == condition)]
                data_for_all_targets_PA.append(np.median(filtered_data_var['Median']))
                data_for_all_targets_PA_min.append(np.min(filtered_data_var['Median']))
                data_for_all_targets_PA_max.append(np.max(filtered_data_var['Median']))
                angles_PA.append(-target * np.pi / 180 + np.pi * 0.5)
                cols_PA.append(colors_PA[m])
                #legs_PA.append(legends_PA[m] + ' ' + str(np.round(np.ptp(np.exp(
                #    filtered_data_condition['Median']) + min_values[n] - 1), 2)) + unit_measure[n])
                legs_PA.append(legends_PA[m])
        
        # POLAR PLOTS FOR CI CONDITIONS
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax.set_rlabel_position(270)
        data_for_all_targets_CI += data_for_all_targets_CI[:1]
        data_for_all_targets_CI_min += data_for_all_targets_CI_min[:1]
        data_for_all_targets_CI_max += data_for_all_targets_CI_max[:1]
        angles_CI += angles_CI[:1]
        legs_CI += legs_CI[:1]
        cols_CI += cols_CI[:1]
        scaler_CI = StandardScaler()
        combined_data = np.concatenate([data_for_all_targets_CI_min, data_for_all_targets_CI,
                                        data_for_all_targets_CI_max]).reshape(-1, 1)
        scaler_CI.fit(combined_data)
        data_for_all_targets_CI_normalized = scaler_CI.transform(np.array(data_for_all_targets_CI).reshape(-1, 1))
        data_for_all_targets_CI_normalized = np.asarray(data_for_all_targets_CI_normalized).flatten()
        #ax.scatter(angles_CI, data_for_all_targets_CI_normalized, color=cols_CI)
        ax.scatter(angles_CI[0:-1:3], data_for_all_targets_CI_normalized[0:-1:3], label=legs_CI[0], color=cols_CI[0], edgecolors='black')
        ax.scatter(angles_CI[1:-1:3], data_for_all_targets_CI_normalized[1:-1:3], label=legs_CI[1], color=cols_CI[1], marker='s', edgecolors='black')
        ax.scatter(angles_CI[2:-1:3], data_for_all_targets_CI_normalized[2:-1:3], label=legs_CI[2], color=cols_CI[2], marker='*', edgecolors='black', s=100)
        data_for_all_targets_CI_min_normalized = scaler_CI.transform(np.array(data_for_all_targets_CI_min).reshape(-1, 1))
        data_for_all_targets_CI_min_normalized = np.asarray(data_for_all_targets_CI_min_normalized).flatten()
        #ax.scatter(angles_CI, data_for_all_targets_CI_min_normalized, color=cols_CI)
        ax.scatter(angles_CI[0:-1:3], data_for_all_targets_CI_min_normalized[0:-1:3], color=cols_CI[0], marker='^', alpha=0.8, edgecolors='black')
        ax.scatter(angles_CI[1:-1:3], data_for_all_targets_CI_min_normalized[1:-1:3], color=cols_CI[1], marker='d', alpha=0.8, edgecolors='black')
        ax.scatter(angles_CI[2:-1:3], data_for_all_targets_CI_min_normalized[2:-1:3], color=cols_CI[2], marker='P', alpha=0.8, edgecolors='black')
        data_for_all_targets_CI_max_normalized = scaler_CI.transform(np.array(data_for_all_targets_CI_max).reshape(-1, 1))
        data_for_all_targets_CI_max_normalized = np.asarray(data_for_all_targets_CI_max_normalized).flatten()
        #ax.scatter(angles_CI, data_for_all_targets_CI_max_normalized, color=cols_CI)
        ax.scatter(angles_CI[0:-1:3], data_for_all_targets_CI_max_normalized[0:-1:3], color=cols_CI[0], marker='v', alpha=0.8, edgecolors='black')
        ax.scatter(angles_CI[1:-1:3], data_for_all_targets_CI_max_normalized[1:-1:3], color=cols_CI[1], marker='D', alpha=0.8, edgecolors='black')
        ax.scatter(angles_CI[2:-1:3], data_for_all_targets_CI_max_normalized[2:-1:3], color=cols_CI[2], marker='X', alpha=0.8, edgecolors='black')
        sorted_indices = np.argsort(angles)
        sorted_angles = np.array(angles)[sorted_indices]
        data_for_all_targets_CI_normalized = np.asarray(data_for_all_targets_CI_normalized)
        sorted_values_1 = data_for_all_targets_CI_normalized[::3]
        sorted_values_1 = np.array(sorted_values_1)[sorted_indices]
        sorted_values_2 = data_for_all_targets_CI_normalized[1::3]
        sorted_values_2 = np.array(sorted_values_2)[sorted_indices]
        sorted_values_3 = data_for_all_targets_CI_normalized[2::3]
        sorted_values_3 = np.array(sorted_values_3)[sorted_indices]
        for i in range(len(angles) - 1):
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_1[i], sorted_values_1[(i + 1)]], color=colors_IC[0])
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_2[i], sorted_values_2[(i + 1)]], color=colors_IC[1])
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_3[i], sorted_values_3[(i + 1)]], color=colors_IC[2])
        result_degree = np.round(-(np.asarray(angles_CI) - np.pi * 0.5) * 180 / np.pi).astype(int)
        result_with_degree = [f"{value}°" for value in result_degree]
        ax.set_xticks(np.asarray(angles_CI))
        ax.set_xticklabels(result_with_degree)
        ax.tick_params(axis='x', labelsize=20)
        angle = np.deg2rad(225)
        ax.legend(loc='center', bbox_to_anchor=(0.05, -0.01), fontsize=20)
        #ax.legend(loc='best', fontsize=20)
        ax.set_title(f'{var_names[n]}', va='bottom', fontsize=30)
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed')
        ytick_labels = np.round(np.exp(data_for_all_targets_CI) + min_values[n] - 1, 3)
        ytick_labels.sort()
        y_ticks = np.sort(data_for_all_targets_CI_normalized)
        if var == 'Head_rotation' or var == 'Signed_error':
            y_ticks = [y_ticks[0],0, y_ticks[-1]]
            y_labels = [str(ytick_labels[0]) + unit_measure[n], '0°', str(ytick_labels[-1]) + unit_measure[n]]
        else:
            y_ticks = [y_ticks[0], y_ticks[-1]]
            y_labels = [str(ytick_labels[0]) + unit_measure[n], str(ytick_labels[-1]) + unit_measure[n]]
        ax.set_yticks(y_ticks)
        ytick_labels = ax.set_yticklabels(y_labels, zorder=2.5, fontsize=20)
        for label, position in zip(ytick_labels, y_ticks):
            # Adjust the radial distance (e.g., 0.2 units away from the circle)
            label.set_position((np.radians(0), position - 0.2))
            if var == 'Head_rotation' or var == 'Signed_error':
                ytick_labels[0].set_position((np.radians(-30), position - 0.2))
                ytick_labels[1].set_position((np.radians(0), position - 0.2))
                ytick_labels[2].set_position((np.radians(30), position - 0.2))
        plot_filename = f'Polar_CI_{var}.png'
        plt.savefig(plot_filename)
        plt.close()

        # POLAR PLOTS FOR HA CONDITIONS
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax.set_rlabel_position(270)
        data_for_all_targets_PA += data_for_all_targets_PA[:1]
        data_for_all_targets_PA_min += data_for_all_targets_PA_min[:1]
        data_for_all_targets_PA_max += data_for_all_targets_PA_max[:1]
        angles_PA += angles_PA[:1]
        legs_PA += legs_PA[:1]
        cols_PA += cols_PA[:1]
        scaler_PA = StandardScaler()
        combined_data = np.concatenate([data_for_all_targets_PA_min, data_for_all_targets_PA,
                                        data_for_all_targets_PA_max]).reshape(-1, 1)
        scaler_PA.fit(combined_data)
        data_for_all_targets_PA_normalized = scaler_PA.transform(np.array(data_for_all_targets_PA).reshape(-1, 1))
        data_for_all_targets_PA_normalized = np.asarray(data_for_all_targets_PA_normalized).flatten()
        #ax.scatter(angles_PA, data_for_all_targets_PA_normalized, color=cols_PA, edgecolors='black')
        ax.scatter(angles_PA[0:-1:4], data_for_all_targets_PA_normalized[0:-1:4], label=legs_PA[0], color=cols_PA[0])
        ax.scatter(angles_PA[1:-1:4], data_for_all_targets_PA_normalized[1:-1:4], label=legs_PA[1], color=cols_PA[1], marker='s', edgecolors='black')
        ax.scatter(angles_PA[2:-1:4], data_for_all_targets_PA_normalized[2:-1:4], label=legs_PA[2], color=cols_PA[2], marker='*', edgecolors='black', s=100)
        ax.scatter(angles_PA[3:-1:4], data_for_all_targets_PA_normalized[3:-1:4], label=legs_PA[3], color=cols_PA[3], marker='8', edgecolors='black')
        data_for_all_targets_PA_min_normalized = scaler_PA.transform(np.array(data_for_all_targets_PA_min).reshape(-1, 1))
        data_for_all_targets_PA_min_normalized = np.asarray(data_for_all_targets_PA_min_normalized).flatten()
        #ax.scatter(angles_PA, data_for_all_targets_PA_min_normalized, color=cols_PA, alpha=1)
        ax.scatter(angles_PA[0:-1:4], data_for_all_targets_PA_min_normalized[0:-1:4], color=cols_PA[0], marker='^', alpha=1, edgecolors='black')
        ax.scatter(angles_PA[1:-1:4], data_for_all_targets_PA_min_normalized[1:-1:4], color=cols_PA[1], marker='d', alpha=1, edgecolors='black')
        ax.scatter(angles_PA[2:-1:4], data_for_all_targets_PA_min_normalized[2:-1:4], color=cols_PA[2], marker='P', alpha=1, edgecolors='black')
        ax.scatter(angles_PA[3:-1:4], data_for_all_targets_PA_min_normalized[3:-1:4], color=cols_PA[3], marker='h', alpha=1, edgecolors='black')
        data_for_all_targets_PA_max_normalized = scaler_PA.transform(np.array(data_for_all_targets_PA_max).reshape(-1, 1))
        data_for_all_targets_PA_max_normalized = np.asarray(data_for_all_targets_PA_max_normalized).flatten()
        #ax.scatter(angles_PA, data_for_all_targets_PA_max_normalized, color=cols_PA, marker='v', alpha=1)
        ax.scatter(angles_PA[0:-1:4], data_for_all_targets_PA_max_normalized[0:-1:4], color=cols_PA[0], marker='v', alpha=1, edgecolors='black')
        ax.scatter(angles_PA[1:-1:4], data_for_all_targets_PA_max_normalized[1:-1:4], color=cols_PA[1], marker='D', alpha=1, edgecolors='black')
        ax.scatter(angles_PA[2:-1:4], data_for_all_targets_PA_max_normalized[2:-1:4], color=cols_PA[2], marker='X', alpha=1, edgecolors='black')
        ax.scatter(angles_PA[3:-1:4], data_for_all_targets_PA_max_normalized[3:-1:4], color=cols_PA[3], marker='H', alpha=1, edgecolors='black')
        sorted_indices = np.argsort(angles)
        sorted_angles = np.array(angles)[sorted_indices]
        data_for_all_targets_PA_normalized = np.asarray(data_for_all_targets_PA_normalized)
        sorted_values_1 = data_for_all_targets_PA_normalized[::4]
        sorted_values_1 = np.array(sorted_values_1)[sorted_indices]
        sorted_values_2 = data_for_all_targets_PA_normalized[1::4]
        sorted_values_2 = np.array(sorted_values_2)[sorted_indices]
        sorted_values_3 = data_for_all_targets_PA_normalized[2::4]
        sorted_values_3 = np.array(sorted_values_3)[sorted_indices]
        sorted_values_4 = data_for_all_targets_PA_normalized[3::4]
        sorted_values_4 = np.array(sorted_values_4)[sorted_indices]
        for i in range(len(angles) - 1):
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_1[i], sorted_values_1[(i + 1)]], color=colors_PA[0])
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_2[i], sorted_values_2[(i + 1)]], color=colors_PA[1])
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_3[i], sorted_values_3[(i + 1)]], color=colors_PA[2])
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_4[i], sorted_values_4[(i + 1)]], color=colors_PA[3])
        result_degree = np.round(-(np.asarray(angles_PA) - np.pi * 0.5) * 180 / np.pi).astype(int)
        result_with_degree = [f"{value}°" for value in result_degree]
        ax.set_xticks(np.asarray(angles_PA))
        ax.set_xticklabels(result_with_degree, fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        angle = np.deg2rad(225)
        ax.legend(loc='center', bbox_to_anchor=(0.05, 0.1), fontsize=20)
        ax.set_title(f'{var_names[n]}', va='bottom', fontsize=30)
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed')
        ytick_labels = np.round(np.exp(data_for_all_targets_PA) + min_values[n] - 1, 2)
        ytick_labels.sort()
        y_ticks = np.sort(data_for_all_targets_PA_normalized)
        if var == 'Head_rotation' or var == 'Signed_error':
            y_ticks = [y_ticks[0],0, y_ticks[-1]]
            y_labels = [str(ytick_labels[0]) + unit_measure[n], '0°', str(ytick_labels[-1]) + unit_measure[n]]
        else:
            y_ticks = [y_ticks[0], y_ticks[-1]]
            y_labels = [str(ytick_labels[0]) + unit_measure[n], str(ytick_labels[-1]) + unit_measure[n]]
        ax.set_yticks(y_ticks)
        ytick_labels = ax.set_yticklabels(y_labels, zorder=2.5, fontsize=20)
        for label, position in zip(ytick_labels, y_ticks):
            # Adjust the radial distance (e.g., 0.2 units away from the circle)
            label.set_position((np.radians(0), position - 0.2))
            if var == 'Head_rotation' or var == 'Signed_error':
                ytick_labels[0].set_position((np.radians(-30), position - 0.2))
                ytick_labels[1].set_position((np.radians(0), position - 0.2))
                ytick_labels[2].set_position((np.radians(30), position - 0.2))
        plot_filename = f'Polar_HA_{var}.png'
        plt.savefig(plot_filename)
        plt.close()

        # POLAR PLOTS FOR HA AND CI CONDITIONS
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax.set_rlabel_position(270)
        data_for_all_targets_PA += data_for_all_targets_PA[:1]
        angles_PA += angles_PA[:1]
        legs_PA += legs_PA[:1]
        cols_PA += cols_PA[:1]
        sorted_indices = np.argsort(angles)
        sorted_angles = np.array(angles)[sorted_indices]
        scaler_combined = StandardScaler()
        combined_data = np.concatenate([data_for_all_targets_PA_min, data_for_all_targets_PA,
                                        data_for_all_targets_PA_max, data_for_all_targets_CI_min, 
                                        data_for_all_targets_CI, data_for_all_targets_CI_max]).reshape(-1, 1)
        scaler_combined.fit(combined_data)
        data_for_all_targets_PA_normalized = scaler_combined.transform(np.array(
            data_for_all_targets_PA).reshape(-1, 1)).flatten()
        data_for_all_targets_CI_normalized = scaler_combined.transform(np.array(
            data_for_all_targets_CI).reshape(-1, 1)).flatten()
        data_for_all_targets_PA_min_normalized = scaler_combined.transform(np.array(
            data_for_all_targets_PA_min).reshape(-1, 1)).flatten()
        data_for_all_targets_CI_min_normalized = scaler_combined.transform(np.array(
            data_for_all_targets_CI_min).reshape(-1, 1)).flatten()
        data_for_all_targets_PA_max_normalized = scaler_combined.transform(np.array(
            data_for_all_targets_PA_max).reshape(-1, 1)).flatten()
        data_for_all_targets_CI_max_normalized = scaler_combined.transform(np.array(
            data_for_all_targets_CI_max).reshape(-1, 1)).flatten()
        data_for_all_targets_PA_normalized = np.asarray(data_for_all_targets_PA_normalized)
        sorted_values_2PA = data_for_all_targets_PA_normalized[1::4]
        sorted_values_2PA = np.array(sorted_values_2PA)[sorted_indices]
        ax.scatter(sorted_angles, sorted_values_2PA, label=legs_PA[1], color=cols_PA[1], edgecolors='black')
        data_for_all_targets_PA_min_normalized = np.asarray(data_for_all_targets_PA_min_normalized)
        sorted_values_2PA_min = data_for_all_targets_PA_min_normalized[1::4]
        sorted_values_2PA_min = np.array(sorted_values_2PA_min)[sorted_indices]
        ax.scatter(sorted_angles, sorted_values_2PA_min, color=cols_PA[1], alpha=1, marker='^', s=40, edgecolors='black')
        data_for_all_targets_PA_max_normalized = np.asarray(data_for_all_targets_PA_max_normalized)
        sorted_values_2PA_max = data_for_all_targets_PA_max_normalized[1::4]
        sorted_values_2PA_max = np.array(sorted_values_2PA_max)[sorted_indices]
        ax.scatter(sorted_angles, sorted_values_2PA_max, color=cols_PA[1], alpha=1, marker='v', s=40, edgecolors='black')
        data_for_all_targets_CI_normalized = np.asarray(data_for_all_targets_CI_normalized)
        sorted_values_2CI = data_for_all_targets_CI_normalized[1::3]
        sorted_values_2CI = np.array(sorted_values_2CI)[sorted_indices]
        ax.scatter(sorted_angles, sorted_values_2CI, label=legs_CI[1], color=cols_CI[1], marker='s', edgecolors='black')
        data_for_all_targets_CI_min_normalized = np.asarray(data_for_all_targets_CI_min_normalized)
        sorted_values_2CI_min = data_for_all_targets_CI_min_normalized[1::3]
        sorted_values_2CI_min = np.array(sorted_values_2CI_min)[sorted_indices]
        ax.scatter(sorted_angles, sorted_values_2CI_min, color=cols_CI[1], alpha=1, marker='d', s=40, edgecolors='black')
        data_for_all_targets_CI_max_normalized = np.asarray(data_for_all_targets_CI_max_normalized)
        sorted_values_2CI_max = data_for_all_targets_CI_max_normalized[1::3]
        sorted_values_2CI_max = np.array(sorted_values_2CI_max)[sorted_indices]
        ax.scatter(sorted_angles, sorted_values_2CI_max, color=cols_CI[1], alpha=1, marker='D', s=40, edgecolors='black')
        for i in range(len(angles) - 1):
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_2PA[i], sorted_values_2PA[(i + 1)]], color='g')
            ax.plot([sorted_angles[i], sorted_angles[(i + 1)]], [
                sorted_values_2CI[i], sorted_values_2CI[(i + 1)]], color='y')
        ax.set_xticks(np.asarray(angles_PA))
        result_degree = np.round(-(np.asarray(angles_PA) - np.pi * 0.5) * 180 / np.pi).astype(int)
        result_with_degree = [f"{value}°" for value in result_degree]
        ax.set_xticklabels(result_with_degree, fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.legend(loc='center', bbox_to_anchor=(0.05, -0.01), fontsize=20)
        ax.set_title(f'{var_names[n]}', va='bottom', fontsize=30)
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed')
        ytick_labels = np.round(np.exp(data_for_all_targets_PA) + min_values[n] - 1, 2)
        ytick_labels.sort()
        vals_for_ticks = np.concatenate((sorted_values_2CI, sorted_values_2PA), axis=None)
        y_ticks = np.sort(vals_for_ticks)
        if var == 'Head_rotation' or var == 'Signed_error':
            y_ticks = [y_ticks[0],0, y_ticks[-1]]
            y_labels = [str(ytick_labels[0]) + unit_measure[n], '0°', str(ytick_labels[-1]) + unit_measure[n]]
        else:
            y_ticks = [y_ticks[0], y_ticks[-1]]
            y_labels = [str(ytick_labels[0]) + unit_measure[n], str(ytick_labels[-1]) + unit_measure[n]]
        ax.set_yticks(y_ticks)
        ytick_labels = ax.set_yticklabels(y_labels, zorder=2.5, fontsize=20)
        for label, position in zip(ytick_labels, y_ticks):
            # Adjust the radial distance (e.g., 0.2 units away from the circle)
            label.set_position((np.radians(0), position - 0.2))
            if var == 'Head_rotation' or var == 'Signed_error':
                ytick_labels[0].set_position((np.radians(-30), position - 0.2))
                ytick_labels[1].set_position((np.radians(0), position - 0.2))
                ytick_labels[2].set_position((np.radians(30), position - 0.2))
        plot_filename = f'Polar_HA_CI_{var}.png'
        plt.savefig(plot_filename)
        plt.close()