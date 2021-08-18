from sqlitedict import SqliteDict
from sqlalchemy import Column, Integer, Unicode, UnicodeText, String, Float, BigInteger, Index
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
engine = create_engine('sqlite:///stock.db', echo=False)
Base = declarative_base(bind=engine)

class Record(Base):
    __tablename__ = 'records'
    date = Column(Integer(), primary_key=True)
    open = Column(Float())
    high = Column(Float())
    low = Column(Float())
    close = Column(Float())
    preclose = Column(Float())
    count = Column(Integer())
    datetime = Column(Integer())
    change = Column(Float())
    pctchange = Column(Float())

    turnover = Column(Float())
    vr = Column(Float())
    volume = Column(Float())
    amount = Column(Float())
    code = Column(Unicode(40), primary_key=True)
    freq = Column(Unicode(10), primary_key=True)


    def __init__(self, count,trade_date,ts_code, open, high, low, close, change, vol, amount,datetime, turnover=0, vr=0, freq='D',preclose=0, **kwargs):
        if not preclose:
            self.preclose = close*(1-change/100)
        self.date = int(trade_date)
        self.amount = amount
        self.open = open
        self.high = high
        self.low = low
        self.vr = vr
        self.close = close
        self.preclose = preclose
        self.pctchange = change
        self.count = count
        self.datetime = datetime
        self.change = change
        self.turnover = turnover
        self.volume = vol
        self.code = ts_code
        self.freq = freq


#
# Base.metadata.create_all()
# mymodel_code_index = Index('mymodel_code_idx', Record.code)
# mymodel_date_index = Index('mymodel_date_idx', Record.date)
# mymodel_freq_index = Index('mymodel_freq_idx', Record.freq)
# mymodel_datetime_index = Index('mymodel_datetime_idx', Record.datetime)
#
# mymodel_code_index.create(bind=engine)
# mymodel_date_index.create(bind=engine)
# mymodel_freq_index.create(bind=engine)
# mymodel_datetime_index.create(bind=engine)

Session = sessionmaker(bind=engine)
s = Session()
records = s.query(Record)
c = records.count()
print(c)

