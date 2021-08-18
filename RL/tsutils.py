from db import Record, s
import tushare as ts
ts.set_token('28db2d35438504f8f9c88e6971fe24257d0760e758f238182631134f')
tspro = ts.pro_api()
import pandas as pd
def get_order_book_id_list():
    """获取所有的股票代码列表
    """
    info = tspro.stock_basic()
    print(info.shape)
    info = info[~info.name.str.startswith(('*', 'ST'))]
    print(info.shape)

    code_list = info.ts_code.sort_values().tolist()
    order_book_id_list = [x for x in code_list if str(x).startswith(('00', '60', '30'))]
    return order_book_id_list

def sync_all_data(start, end, freq='W'):
    idlist = get_order_book_id_list()
    print(idlist)
    for code in idlist:
        if code.startswith(('30')):
            continue
        try:

            df = ts.pro_bar(ts_code=code, adj='qfq', start_date=start, end_date=end, freq=freq,factors=['vr', 'tor'])
            print(f"{code} saved")
            df["datetime"] = df["trade_date"].apply(lambda x: int(x.replace("-", "")) * 1000000)

            arr = df.to_dict('records')
            arr = arr[::-1]
            for k, rec in enumerate(arr):
                # print(rec)
                obj = Record(count=k, **rec, freq=freq)
                try:
                    s.merge(obj)
                except Exception as e:
                    print(e)
                    pass
            s.commit()
            res = s.query(Record)
            print(res.count())
        except Exception as e:
            print(code, ' error', e)

def show_bars(code, start, end, freq):
    df = ts.pro_bar(ts_code=code, adj='qfq', start_date=start, end_date=end, freq=freq, factors=['vr', 'tor'])
    print(df.head(5).tolist())
    print(df.shape)

def get_data(code, start=None, end=None, sess=None, freq='D'):
    start = int(start)
    end = int(end)
    arr = pd.DataFrame()
    try:
        if start and end:
            res = sess.query(Record).filter(Record.code==code, Record.date>=start, Record.date<=end, Record.freq==freq).order_by(Record.date)
        else:
            res = sess.query(Record).filter_by(code=code, freq=freq).order_by(Record.date)
        arr = []
        for r in res:
            try:
                datetime = int.from_bytes(r.datetime,'little')
            except:
                datetime = r.datetime

            rd = r.__dict__
            rd['datetime'] = datetime
            rd['date'] = str(r.date)
            del rd['_sa_instance_state']
            arr.append(rd)
            # rec = [r.count, str(r.date), r.openp, r.high, r.low, r.close, r.preclose, r.change,r.pctchange, r.volume, r.amount,dateint]
            # arr.append(rec)
        arr = pd.DataFrame(arr)
        arr = arr.to_records()
    except Exception as e:
        print(e)
    return arr

if __name__ == '__main__':
    # show_bars(start='20180101',end='20210804',freq='60min',code='000001.SZ')

    sync_all_data(start='20190101',end='20210818',freq='D')
    # x = get_data('000001.SZ', 20200101, 20210812, sess=s)
    # print(x)