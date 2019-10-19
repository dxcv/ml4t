import pandas as pd
import datetime as dt
import marketsimcode as ms
from util import get_data


class TheoreticallyOptimalStrategy:
    def __init__(self):
        pass

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=1e6):
        """
            Legal daily trade values are as follows:
                * +1000 (BUY 1000 shares)
                * -1000 (SELL 1000 shares)
                * +2000 (BUY 2000 shares if short 1000)
                * -2000 (SELL 2000 shares if long 1000)
            params:
            - symbol: stock symbol to act on
            - sd: datetime start date
            - ed: datetime end date
            - sv: starting portfolio value
            return:
            - df_trades: data frame whose values represent daily trades
        """
        # get prices for underlying
        pxs = get_data([symbol], pd.date_range(sd, ed))
        pxs = pxs[symbol]
        pxchgs = pxs/pxs.shift(1)-1

        # prior day ups and downs
        pups = pxchgs > 0.0
        pdwns = ~pups

        # forward day ups and downs
        fups = pups.shift(-1).fillna(False)
        fdwns = pdwns.shift(-1).fillna(False)

        # set buys to max shares
        buys = fups & pdwns
        buys[0] = fups[1]
        buys[buys] = 2000

        # set sells to max shares
        sells = ~fups & pups
        sells[0] = fdwns[1]
        sells[sells] = -2000

        # consolidate trades and cvt first to 1000 lot
        trades = buys+sells
        tidx = trades.nonzero()[0][0]
        trades[tidx] = trades[tidx]/2

        return pd.DataFrame(trades)


if __name__ == '__main__':
    opt_strat = TheoreticallyOptimalStrategy()
    df_trades = opt_strat.testPolicy()
    print(f'Theoretically Optimal Strategy:\n{df_trades.head(10)}')
    portvals = ms.compute_portvals(df_trades, start_val=1e6,
                                   commission=0.0, impact=0.0)
    print(f'Portfolio Values:\n{portvals.head()}')
