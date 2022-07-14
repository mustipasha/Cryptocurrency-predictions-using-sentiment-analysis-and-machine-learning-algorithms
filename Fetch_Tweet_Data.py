import twint
import nest_asyncio


nest_asyncio.apply()
# Configure
c = twint.Config()
c.Search = "USDT Tether #usdt #tether"
c.Hashtags = True
c.Since = "2022-05-15"
c.Until = "2022-06-15"
c.Store_csv = True
c.Output = "twitter_USDT_06_15.csv"

# Run
twint.run.Search(c)



