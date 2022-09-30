import pandas as pd
import numpy as np
import statsmodels.api as sm

class port_an:



    def __init__(self, returns,ann_factor, benchmark=None, rf=None ):
        self.returns = returns
        self.ann_factor = ann_factor


        if rf is None:
            rf = pd.DataFrame(data = 0, index = returns.index, columns = ['rf'])

        else:
            rf.columns = ['rf']
        if benchmark is None:
            returns_all = returns

        else:
            self.benchmark = benchmark.sub(rf['rf'], axis=0)
            returns_all = pd.concat([returns, benchmark], axis=1)

        xsReturns = returns_all
        self.xsReturns = xsReturns


        nPeriods = len(returns_all)
        self.nPeriods = nPeriods

        NAVs = (1 + returns_all).cumprod()
        self.NAVs = NAVs

        gAvg = (1 + returns_all).prod() ** (ann_factor / nPeriods) - 1
        self.gAvg = gAvg
        gAvg_xs = (1 + xsReturns).prod() ** (ann_factor / nPeriods) - 1
        self.gAvg_xs = gAvg_xs

        vol = xsReturns.std() * np.sqrt(ann_factor)
        SR = gAvg_xs / vol

        self.SR = SR
        self.vol = vol
        self.skew_strat = returns_all.skew()
        self.kurtosis_strat = returns_all.kurtosis()
        self.alphas = list()
        self.betas = list()
        if benchmark is None:
            benchmark = 0
        else:
            for strat in xsReturns.columns:

                OLS_model = sm.OLS(xsReturns[strat], sm.add_constant(self.benchmark)).fit()
                self.alphas.append(OLS_model.params[0])
                self.betas.append(OLS_model.params[1])

    def an_w_BM(self):
        self.IR = (self.gAvg_xs.iloc[0:-1] - self.gAvg_xs.iloc[-1]) / self.vol
        df_summary = pd.DataFrame(index = self.xsReturns.keys(), columns =['Geometric Average - Excess (%)', 'Volatility Annual (%)', 'Sharpe Ratio', 'Information Ratio', 'Skewness', 'Excess Kurtosis', 'Beta', 'Alpha (%)'])
        df_summary.iloc[:, 0] = round(self.gAvg_xs * 100,2)
        df_summary.iloc[:, 1] = round(self.vol * 100, 2)
        df_summary.iloc[:, 2] = round(self.SR,2)
        df_summary.iloc[:, 3] = round(self.IR , 2)
        df_summary.iloc[:, 4] = round(self.skew_strat, 2)
        df_summary.iloc[:, 5] = round(self.kurtosis_strat, 2)
        df_summary.iloc[:, 6] = self.betas
        df_summary.iloc[:, 6] = round(df_summary.iloc[:, 6],2)
        df_summary.iloc[:, 7] = self.alphas
        df_summary.iloc[:, 7] = round(df_summary.iloc[:, 7]*100,2)
        return df_summary.fillna(0)

    def an_no_BM(self):
        df_summary = pd.DataFrame(index = self.xsReturns.keys(), columns =['Geometric Average - Excess (%)', 'Volatility Annual (%)', 'Sharpe Ratio', 'Skewness', 'Excess Kurtosis', 'Beta', 'Alpha (%)'])
        df_summary.iloc[:, 0] = round(self.gAvg_xs * 100,2)
        df_summary.iloc[:, 1] = round(self.vol * 100, 2)
        df_summary.iloc[:, 2] = round(self.SR,2)
        df_summary.iloc[:, 3] = round(self.skew_strat, 2)
        df_summary.iloc[:, 4] = round(self.kurtosis_strat, 2)

        return df_summary.fillna(0)



