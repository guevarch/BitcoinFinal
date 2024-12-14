# Import Required Modules
from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px
import pandas_ta as pta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import date
from datetime import timedelta, date
from prophet import Prophet
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import mpld3
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib import colors
import plotly.io as pio
import plotly.subplots as sp
import warnings
# Create Home Page Route
app = Flask(__name__)


@app.route('/')
def bar_with_plotly():

	df = pd.read_csv("Resources/btcjoin.csv", parse_dates=['date'])
	btc_df = yf.Ticker('BTC-USD').history(period='7y',interval='1d',actions=False).reset_index()
	btc_df = btc_df.loc[(btc_df['Date'] > '2022-10-25')]
	btc_df['Close']=btc_df['Close'].astype("float")
	df['price']=df['price'].str.replace(',','')
	df['price']=df['price'].astype("float")
	btc_df = btc_df.rename(columns={"Close": "price", "Date":"date"})
	btc_df['date'] = btc_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
	btc_df['date'] = pd.to_datetime(btc_df['date'])

	df = pd.merge(df, btc_df, on=['date', 'price'], how='outer')
	df = pd.merge(df, btc_df, on=['date', 'price'], how='outer')
	df = df.rename(columns={"value": "wallets"})
	df = df.drop(columns=['volume','change', 'low', 'high', 'open','Unnamed: 0', "wallets", "address", "mined"])
	df['200D'] = df['price'].rolling(200).mean()
	df['300D'] = df['price'].rolling(300).mean()
	df['50D'] = df['price'].rolling(50).mean()
	df['7D'] = df['price'].rolling(7).mean()
	# df = df.dropna()
	df['meanavge'] = (df['200D'] + df['300D'] + df['50D'] )/3
	# df = df.drop(columns=['200D','300D', '50D'])
	df['meanvalue'] = df["price"] - df["meanavge"]
	df['status'] = df['meanvalue'].apply(lambda x: '1' if x > 0 else '0')
	df['status']=df['status'].astype("object")
	df['price-meanavge']=df['price'] - df['meanavge']
	df['move%'] = (df['price-meanavge']/(df['price'] + df['meanavge']))
	bins = [-.43, -.18, 0, .18, .43]
	group_names = ["2 STDV Under","1 STDV Under", "1 STDV Over","2 STDV Over "]
	df["Valuation"] = pd.cut(df["move%"], bins, labels=group_names)

	k = df['price'].ewm(span=12, adjust=False, min_periods=12).mean()

	# Get the 12-day EMA of the closing price
	d = df['price'].ewm(span=26, adjust=False, min_periods=26).mean()

	# Subtract the 26-day EMA from the 12-Day EMA to get the MACD
	macd = k - d

	# Get the 9-Day EMA of the MACD for the Trigger line
	macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()

	# Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
	macd_h = macd - macd_s

	# Add all of our new values for the MACD to the dataframe
	df['macd'] = df.index.map(macd)
	df['macd_h'] = df.index.map(macd_h)
	df['macd_s'] = df.index.map(macd_s)


	df['priceL'] = np.log(df['price'])

	df_train = df[['date', 'priceL']]
	df_train = df_train.rename(columns = {"date":"ds", "priceL":"y"})

	# instantiate the model and set parameters
	model = Prophet()

	# fit the model to historical data
	model.fit(df_train);

	start = "2010-09-25"
	end = date.today() + timedelta(days=60)
	insample = pd.DataFrame(pd.date_range(start,end, periods=92))

	# Change the column name
	insample.columns = ['ds']

	# in-sample prediction
	prediction = model.predict(insample)


	# Buy Zones
	fig = px.scatter(df, x="date", y="price", color="Valuation", color_discrete_sequence=["red","green","blue","orange"],
                 title="price")
	fig.add_trace(go.Scatter(name="MeanAvg", x=df['date'], y=df['meanavge'], marker = {'color' : 'black'}, legendrank=2))
	fig.add_trace(go.Scatter(x=prediction['ds'], y=np.exp(prediction['yhat']),
		fill=None,
		mode='lines',
		line_color='lightblue',
		))
	fig.add_trace(go.Scatter(
		x=prediction['ds'],
		y=np.exp(prediction['yhat_lower']),
		fill='tonexty', # fill area between trace0 and trace1
		mode='lines', line_color='lightblue'))

	fig.add_trace(go.Scatter(
		x=prediction['ds'],
		y=np.exp(prediction['yhat_upper']),
		fill='tonexty', # fill area between trace0 and trace1
		mode='lines', line_color='lightblue'))

	fig.update_yaxes(fixedrange=False)
	fig.update_layout(title_text='Bitcoin Prophet Model + Buy Zones')
	fig.update_yaxes(type="log")
	fig.update_xaxes(ticklabelposition="inside top", title="Date")
	fig.update_yaxes(nticks=12)
	fig.update_xaxes(nticks=50)
	fig.update_layout(
		margin=dict(l=20, r=100, t=70, b=20),
	)
	fig.update_layout(height=500, width=1000)
	fig.update_layout(showlegend=False)
	fig.add_vline(x='2012-11-28', line_width=3, line_dash="dash", line_color="green")
	fig.add_vline(x='2016-07-09', line_width=3, line_dash="dash", line_color="green")	
	fig.add_vline(x='2020-05-11', line_width=3, line_dash="dash", line_color="green")	
	fig.add_vline(x='2024-04-20', line_width=3, line_dash="dash", line_color="green")
	fig.update_layout(template='plotly_white')
 
	# Create figure
	# Define colors
	colors = ["red", "orange", "blue", "green"]

	# Create figure
	fig.add_trace(go.Indicator(
		mode="number+gauge+delta",
		value=df['move%'].iloc[-1],
		delta={'reference': 0},
		title={'text': df['Valuation'].iloc[-1]},
		domain={'x': [0.5, .9], 'y': [0.1, 0.18]},
		gauge={
			'shape': "bullet",
			'axis': {'range': [min(bins), max(bins)]},
			'bar': {'color': 'black'},
			'steps': [{'range': [bins[i], bins[i + 1]], 'color': colors[i]} for i in range(len(bins) - 1)]
		}
	))
	Buyzones = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	fig = px.scatter(df, x="date", y="price", color="Valuation", color_discrete_sequence=["red","green","blue","orange"],
                 title="price")
	fig.add_trace(go.Scatter(name="200D", x=df['date'], y=df['200D'], marker = {'color' : 'red'}, legendrank=2))
	fig.add_trace(go.Scatter(name="MeanAverage", x=df['date'], y=df['meanavge'], marker = {'color' : 'purple'}, legendrank=2))
	fig.add_trace(go.Scatter(name="50D", x=df['date'], y=df['50D'], marker = {'color' : 'blue'}, legendrank=2))
	fig.add_trace(go.Scatter(name="300D", x=df['date'], y=df['300D'], marker = {'color' : 'white'}, legendrank=2))

	# Moving Averages
	fig.update_yaxes(fixedrange=False)
	fig.update_layout(title_text='Moving Averages')
	fig.update_yaxes(type="log")
	fig.update_xaxes(ticklabelposition="inside top", title="Date")
	fig.update_yaxes(nticks=12)
	fig.update_xaxes(nticks=50)
	fig.update_layout(margin=dict(l=20, r=100, t=70, b=20))
	fig.update_layout(height=500, width=1000)
	fig.update_layout(showlegend=True)
	fig.add_vline(x='2012-11-28', line_width=3, line_dash="dash", line_color="green")
	fig.add_vline(x='2016-07-09', line_width=3, line_dash="dash", line_color="green")	
	fig.add_vline(x='2020-05-11', line_width=3, line_dash="dash", line_color="green")	
	fig.add_vline(x='2024-04-20', line_width=3, line_dash="dash", line_color="green")
	fig.update_layout(template='plotly_white')
	Movingaverages = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	



	#Indicators
	fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    title = {'text': "Current Valuation - Standard Deviation percent move from meanaverage", 'font': {'size': 15}},
    value = df['move%'].iloc[-1],
    domain = {'x': [0,1], 'y': [0,1]},
    gauge = {
        'axis': {'range': [-0.43, 0.43], 'tickwidth': 1, 'tickcolor': "orange"},
        'bar': {'color': "orange"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "white",
        'steps': [
            {'range': [-0.43, -0.18], 'color': 'red'},
            {'range': [-0.18, 0], 'color': 'yellow'},
            {'range': [0, 0.18], 'color': 'blue'},
            {'range': [0.18,0.43], 'color': 'green'}],
        }))

	fig.update_layout(
		font={'color': "black", 'family': "Arial"},
		xaxis={'showgrid': False, 'range':[-1,1]},
		yaxis={'showgrid': False, 'range':[0,1.11]},
		plot_bgcolor='rgba(0,0,0,0)'
		)
	# Set the visibility ON
	fig.update_yaxes(title='', visible=True, showticklabels=False)
	# Set the visibility OFF
	fig.update_xaxes(title='x', visible=False, showticklabels=False)
	fig.update_layout(height=500, width=1000, title_text="Current Valuation")
	Indicators = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	# YTD

	def get_ytd_returns(ticker):
		df = yf.Ticker(ticker).history(period='ytd',interval='1d',actions=False).reset_index()
		ytd_return = ((df.Close.iloc[-1]/df.Close.iloc[0])-1)
		return ytd_return

	tickers = ['BTC-USD','MSTR', 'GC=F', 'Cl=F', 'TLT', 'DX=F', '^RUT', '^GSPC', '^IXIC', '^DJI', 'BIGPX']
	asset_names = ['Bitcoin','MSTR', 'Gold', 'Oil', '20yr Treasury', 'DXY', 'Russell 2000 Index', 'S&P 500', 'NASDAQ', 'Dow Jones', '60/40']

	data = [[asset_name, get_ytd_returns(ticker)*100] for asset_name, ticker in zip(asset_names, tickers)]

	# Create the pandas DataFrame
	data = pd.DataFrame(data, columns=['Asset', 'YTD%'])
	data['YTD%'] = data['YTD%'].round(2)

	fig = px.bar(data, x='YTD%', y='Asset', orientation='h', text=str("YTD%"), color="YTD%",color_continuous_scale=px.colors.sequential.Oryel)
	fig.layout.yaxis.tickformat = ',.%'
	fig.update_layout(title_text='Year To Date Returns')
	fig.update_layout(yaxis=dict(showticklabels=True))
	fig.update_layout(height=500, width=1000, title_text="YTD Returns")
	fig.update_layout(template='plotly_dark')
	YTD = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	# Corr

	tickers_list = ['BTC-USD','DX=F', '^GSPC', '^IXIC', 'GC=F']

	# Fetch data for all tickers using a for loop
	data = {}
	for ticker in tickers_list:
		data[ticker] = yf.Ticker(ticker).history(period='3mo', interval='1d', actions=False)['Close']

	# Filter out NaN values using list comprehension
	data = {ticker: close[~close.isna()] for ticker, close in data.items()}

	# Merge data using inner join
	merge = pd.concat(data, axis=1, join='inner')
	merge.columns = [ticker for ticker, close in data.items()]

	# Rename columns
	merge = merge.rename(columns={"BTC-USD": "BTC", "DX=F": "DXY", "^GSPC": "S&P500", "^IXIC": "Nasdaq", "GC=F": "Gold"})
	assets = [('DXY', 'DXY/BTC'), ('S&P500', 'S&P500/BTC'), ('Nasdaq', 'Nasdaq/BTC'), ('Gold', 'Gold/BTC')]

	# Use a for loop to calculate the correlation for each asset pair
	correlations = {}
	for asset, col_name in assets:
		correlations[col_name] = merge['BTC'].rolling(3).corr(merge[asset])

	# Use the assign method to add the correlations as new columns to the DataFrame
	merge = merge.assign(**correlations).dropna().reset_index()
	corr_matrix=merge.drop(columns=['DXY/BTC','S&P500/BTC', 'Nasdaq/BTC', 'Gold/BTC'])
	corr_matrix = corr_matrix.corr().round(2)
	corr_matrix = px.imshow(corr_matrix,color_continuous_scale='Oryel', text_auto=True, aspect="auto")
	corr_matrix.update_layout(height=400, width=950, title_text="Correlation Matrix")
	corr1 = json.dumps(corr_matrix, cls=plotly.utils.PlotlyJSONEncoder)
	corr_matrix.write_html("static/corr1.html")
	merge=merge.tail(30)
	fig = make_subplots(
		rows=4, cols=1,start_cell="bottom-left",shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.04, 0.04,0.04,0.04],
		subplot_titles=("DXY/BTC", "S&P500/BTC", "Nasdaq/BTC", "Gold/BTC"))

	fig.add_trace(go.Bar(x=merge['Date'], y=merge['DXY/BTC'],
						marker=dict(color=merge['DXY/BTC'], coloraxis="coloraxis")),
				1, 1)
	fig.add_trace(go.Bar(x=merge['Date'], y=merge['S&P500/BTC'],
						marker=dict(color=merge['S&P500/BTC'], coloraxis="coloraxis")),
				2, 1)

	fig.add_trace(go.Bar(x=merge['Date'], y=merge['Nasdaq/BTC'],
						marker=dict(color=merge['Nasdaq/BTC'], coloraxis="coloraxis")),
				3, 1)

	fig.add_trace(go.Bar(x=merge['Date'], y=merge['Gold/BTC'],
						marker=dict(color=merge['Gold/BTC'], coloraxis="coloraxis")),
				4, 1)
	fig.update_layout(
		margin=dict(l=20, r=20, t=70, b=20),
	)
	fig.update_layout(coloraxis=dict(colorscale='Oryel'), showlegend=False)
	fig.update_yaxes(nticks=3)
	fig.update_xaxes(nticks=25)
	fig.update_layout(height=500, width=1000, title_text="30 Day Correlation")
	fig.update_layout(template='plotly_dark')
	corr2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	# Moving Averages

	fig = go.Figure()

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		number = {'prefix': "$"},
		value = df['200D'].iloc[-1],
		delta = {"reference": ((df['200D'].iloc[-2])), "valueformat": ".0f"},
		title = {"text": "200 Day Moving Average"},
		gauge = {
			'axis': {'visible': False}},
		domain = {'row': 0, 'column': 0}))

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		number = {'prefix': "$"},
		value = df['meanavge'].iloc[-1],
		delta = {"reference": ((df['meanavge'].iloc[-2])), "valueformat": ".0f"},
		title = {"text": "Meanaverage"},
		gauge = {
			'axis': {'visible': False}},
		domain = {'row': 1, 'column': 0}))


	fig.add_trace(go.Indicator(
		mode = "number+delta",
		number = {'prefix': "$"},
		value = df['50D'].iloc[-1],
		delta = {"reference": ((df['50D'].iloc[-2])), "valueformat": ".0f"},
		title = {"text": "50 Day Moving Average"},
		domain = {'row': 0, 'column': 1}))

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		number = {'prefix': "$"},
		value = df['300D'].iloc[-1],
		delta = {"reference": ((df['300D'].iloc[-2])), "valueformat": ".0f"},
		title = {"text": "300 Day Moving Average"},
		domain = {'row': 1, 'column': 1}))

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		number = {'prefix': "$"},
		value = df['price'].iloc[-1],
		delta = {"reference": ((df['price'].iloc[-2])), "valueformat": ".0f"},
		title = {"text": "Current Price"},
		domain = {'row': 0, 'column': 2}))

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		number = {'prefix': "$"},
		value = df['7D'].iloc[-1],
		delta = {"reference": ((df['7D'].iloc[-2])), "valueformat": ".0f"},
		title = {"text": "7 Day Moving Average"},
		domain = {'row': 1, 'column': 2}))

	fig.update_layout(
		grid = {'rows': 2, 'columns': 3, 'pattern': "independent"},
	)
	fig.update_layout(height=500, width=1000, title_text="Moving Averages")
	fig.update_layout(template='plotly_dark')
	Movingaverages2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

 
	# Heatmap

	df["month"] = df["date"].dt.strftime('%Y-%m')

	# Compute the monthly returns
	df_monthly = df.groupby("month")["price"].last().pct_change()

	# Convert the resulting Series to a DataFrame
	df_returns = df_monthly.to_frame(name="returns")

	# Use pivot_table to reshape the data by year and month
	df_pivot = df_returns.pivot_table(index=df_returns.index.str.slice(0,4),
									columns=df_returns.index.str.slice(5,),
									values="returns")

	# import pandas as pd
	df_pivot.rename(columns={'01': 'January', 
							'02': 'February', 
							'03': 'March', 
							'04': 'April', 
							'05': 'May', 
							'06': 'June', 
							'07': 'July', 
							'08': 'August', 
							'09': 'September', 
							'10': 'October', 
							'11': 'November', 
							'12': 'December'}, inplace=True)
	# Create a heatmap using Seaborn

	lighter_red = '#FFA07A'
	lighter_green = '#98FB98'

	heatmap = sns.heatmap(df_pivot, cmap=[lighter_red, lighter_green], cbar=False, annot=True, fmt=".2%", center=0)
	heatmap.tick_params(axis='x', rotation=0)

	# Get the current tick locations and labels of the x-axis
	x_ticks = heatmap.get_xticks()

	# Create a list of all the unique x values in the pivot table
	x_values = df_pivot.columns.get_level_values(0)

	# Set the label for the x-axis of the heatmap
	heatmap.set_xlabel('Month')
	heatmap.set_ylabel('Years')

	# change the figure size
	fig = heatmap.get_figure()
	fig.set_size_inches(14,10) # adjust the width and height as per your preference
	# Set the tick labels for the x-axis
	heatmap.set_xticklabels(x_values)

	# Add a title to the heatmap
	heatmap.set_title("Bitcoin Returns by Month")
	# Show the plot
	heatmap.figure.savefig("static/heatmap.png")
	plt.show()


	# Rainbow + Prophet

	# Buy Zones

	df =  pd.DataFrame(df).reset_index()

	df['date'] = pd.to_datetime(df['date']) # Ensure that the date is in datetime or graphs might look funny
	df = df[df["price"] > 0] # Drop all 0 values as they will fuck up the regression bands

	# this is your log function
	def logFunc(x,a,b,c):
		return a*np.log(b+x) + c

	# getting your x and y data from the dataframe
	xdata = np.array([x+1 for x in range(len(df))])
	ydata = np.log(df["price"])

	# here we ar fitting the curve, you can use 2 data points however I wasn't able to get a graph that looked as good with just 2 points.
	popt, pcov = curve_fit(logFunc, xdata, ydata, p0 = [10,100,90]) # p0 is justa guess, doesn't matter as far as I know

	# This is our fitted data, remember we will need to get the ex of it to graph it
	fittedYData = logFunc(xdata, popt[0], popt[1], popt[2])
	fig = px.scatter(df, x="date", y="price", color="Valuation", color_discrete_sequence=["red","green","blue","orange"],
					title="price")
	fig.add_trace(go.Scatter(name="MeanAvg", x=df['date'], y=df['meanavge'], marker = {'color' : 'black'}, legendrank=2))
	fig.add_trace(go.Scatter(x=prediction['ds'], y=np.exp(prediction['yhat']),
		fill=None,
		mode='lines',
		line_color='lightblue',
		))
	fig.add_trace(go.Scatter(
		x=prediction['ds'],
		y=np.exp(prediction['yhat_lower']),
		fill='tonexty', # fill area between trace0 and trace1
		mode='lines', line_color='lightblue'))

	fig.add_trace(go.Scatter(
		x=prediction['ds'],
		y=np.exp(prediction['yhat_upper']),
		fill='tonexty', # fill area between trace0 and trace1
		mode='lines', line_color='lightblue'))

	fig.update_yaxes(fixedrange=False)
	fig.update_layout(title_text='Bitcoin Prophet Model + Buy Zones')
	fig.update_yaxes(type="log")
	fig.update_xaxes(ticklabelposition="inside top", title="Date")
	fig.update_yaxes(nticks=12)
	fig.update_xaxes(nticks=50)
	fig.update_layout(
		margin=dict(l=20, r=100, t=70, b=20),
	)
	fig.update_layout(height=500, width=1000)
	fig.update_layout(showlegend=False)
	fig.add_vline(x='2012-11-28', line_width=3, line_dash="dash", line_color="green")
	fig.add_vline(x='2016-07-09', line_width=3, line_dash="dash", line_color="green")	
	fig.add_vline(x='2020-05-11', line_width=3, line_dash="dash", line_color="green")	
	fig.add_vline(x='2024-04-20', line_width=3, line_dash="dash", line_color="green")

	for i in range(0, 5):
		fitted_data = np.exp(fittedYData + i * 0.455)
		fig.add_trace(go.Scatter(x=df["date"], y=fitted_data, mode='lines', showlegend=False))

		# You can use the below fill between trace rather than the above line trace, I prefer the line graph
		fig.add_trace(go.Scatter(x=df["date"], y=np.exp(fittedYData + i * .45 - 1), fill=None, mode='lines',
								line=dict(width=0), showlegend=False))
		fig.add_trace(go.Scatter(x=df["date"], y=np.exp(fittedYData + i * .45), fill='tonexty', mode='lines',
								name=f"Rainbow Band {i + 1}", line_color=f"hsl({i * 35},80%,50%)", showlegend=False))
	fig.update_layout(template='plotly_white')
	Rainbow = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


	# Black Rainbow

	# Set dark background theme
	pio.templates.default = "plotly_dark"

	# Create plot with larger size
	fig = go.Figure()
	fig.update_layout(title='Bitcoin Rainbow Chart', xaxis_title='Time', yaxis_title='Bitcoin price in log scale')

	fig.update_layout(
		autosize=False,
		width=1000,
		height=600)

	fig.update_yaxes(type="log")
	fig.add_trace(
		go.Scatter(x=list(df.date), y=list(df.price), showlegend=False, line_color='white'))
	# Draw the rainbow bands
	for i in range(0, 5):
		fitted_data = np.exp(fittedYData + i * 0.455)
		fig.add_trace(go.Scatter(x=df["date"], y=fitted_data, mode='lines', showlegend=False))

		# You can use the below fill between trace rather than the above line trace, I prefer the line graph
		fig.add_trace(go.Scatter(x=df["date"], y=np.exp(fittedYData + i * .45 - 1), fill=None, mode='lines',
								line=dict(width=0), showlegend=False))
		fig.add_trace(go.Scatter(x=df["date"], y=np.exp(fittedYData + i * .45), fill='tonexty', mode='lines',
								name=f"Rainbow Band {i + 1}", line_color=f"hsl({i * 35},80%,50%)", showlegend=False))
		BRainbow = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  
	fig = px.scatter(df, x="date", y="price", color="Valuation", color_discrete_sequence=["red","green","blue","orange"], 
                 title="price")

	# Moving Average Cloud
	fig.add_trace(go.Scatter(name="200D", x=df['date'], y=df['200D'], marker={'color': 'red'}, legendrank=2))
	fig.add_trace(go.Scatter(name="50D", x=df['date'], y=df['50D'], marker={'color': 'blue'}, legendrank=2))
	
	# Adding the clouds
	upper_bound = df['200D'] * 1.3  # 1% above the 200-day MA
	lower_bound = df['50D'] * 0.7   # 1% below the 50-day MA

	fig.add_trace(go.Scatter(x=df['date'], y=upper_bound, line=dict(width=0), showlegend=False))
	fig.add_trace(go.Scatter(x=df['date'], y=lower_bound, line=dict(width=0), fill='tonexty', 
							fillcolor='rgba(245, 166, 35, 0.3)', showlegend=False))  # Orange cloud

	# Note: Adjust the upper_bound and lower_bound calculations according to your needs.
	# The rgba color format includes an alpha value (transparency) where 1 is opaque and 0 is fully transparent.

	# Further styling as per your current setup...
	fig.update_yaxes(fixedrange=False)
	fig.update_layout(title_text='Moving Averages Cloud')
	fig.update_yaxes(type="log")
	fig.update_xaxes(ticklabelposition="inside top", title="Date")
	fig.update_yaxes(nticks=12)
	fig.update_xaxes(nticks=50)
	fig.update_layout(margin=dict(l=20, r=100, t=70, b=20))
	fig.update_layout(height=500, width=1000)
	fig.update_layout(showlegend=True)
	# Vertical lines
	fig.add_vline(x='2012-11-28', line_width=3, line_dash="dash", line_color="green")
	fig.add_vline(x='2016-07-09', line_width=3, line_dash="dash", line_color="green")    
	fig.add_vline(x='2020-05-11', line_width=3, line_dash="dash", line_color="green")    
	fig.add_vline(x='2024-04-20', line_width=3, line_dash="dash", line_color="green")
	fig.update_layout(template='plotly_white')

	cloud = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
 
 
	## Buyzonesbar
	fig = px.bar(df, x="date", y="price", color="Valuation", color_discrete_sequence=["red","green","blue","orange"],
					title="price")
	fig.add_trace(go.Scatter(name="MeanAvg", x=df['date'], y=df['meanavge'], marker = {'color' : 'black'}, legendrank=2))

	fig.update_yaxes(fixedrange=False)
	fig.update_layout(title_text='Bitcoin Buy Zones 2.0')
	fig.update_yaxes(type="log")
	fig.update_xaxes(ticklabelposition="inside top", title="Date")
	fig.update_yaxes(nticks=12)
	fig.update_xaxes(nticks=50)
	# fig.update_layout(
	# 	margin=dict(l=20, r=100, t=70, b=20),
	# )
	fig.update_layout(showlegend=True)
	fig.update_traces(dict(marker_line_width=.01))
	fig.add_vline(x='2012-11-28', line_width=3, line_dash="dash", line_color="green")
	fig.add_vline(x='2016-07-09', line_width=3, line_dash="dash", line_color="green")	
	fig.add_vline(x='2020-05-11', line_width=3, line_dash="dash", line_color="green")	
	fig.add_vline(x='2024-04-20', line_width=3, line_dash="dash", line_color="green")	
	fig.layout.template = 'seaborn'
	Buyzonesbar = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	# Halving Tracker
 
	df['daily_pct_change'] = df['price'].pct_change() * 100
	df = df[['date', 'price', 'daily_pct_change']]
	# Convert the dates to datetime objects
	date1 = pd.to_datetime('2012-11-28')
	date2 = pd.to_datetime('2016-07-9')
	date3 = pd.to_datetime('2020-05-11')
	date4 = pd.to_datetime('2024-04-20')

	# Create the cycle DataFrames
	cycle1 = df[df['date'] <= date1]
	cycle2 = df[(df['date'] > date1) & (df['date'] <= date2)]
	cycle3 = df[(df['date'] > date2) & (df['date'] <= date3)]
	cycle4 = df[(df['date'] > date3) & (df['date'] <= date4)]
	cycle5 = df[df['date'] > date4]

	cycle1 = cycle1.reset_index(drop=True)
	cycle2 = cycle2.reset_index(drop=True)
	cycle3 = cycle3.reset_index(drop=True)
	cycle4 = cycle4.reset_index(drop=True)
	cycle5 = cycle5.reset_index(drop=True)


	cycle1['cumulative_pct_change'] = cycle1['daily_pct_change'].cumsum()
	cycle2['cumulative_pct_change'] = cycle2['daily_pct_change'].cumsum()
	cycle3['cumulative_pct_change'] = cycle3['daily_pct_change'].cumsum()
	cycle4['cumulative_pct_change'] = cycle4['daily_pct_change'].cumsum()
	cycle5['cumulative_pct_change'] = cycle5['daily_pct_change'].cumsum()

	cycle1 = cycle1.reset_index()
	cycle2 = cycle2.reset_index()
	cycle3 = cycle3.reset_index()
	cycle4 = cycle4.reset_index()
	cycle5 = cycle5.reset_index()

	cycle1 = cycle1.rename(columns={'index': 'days from halving'})

	cycle2 = cycle2.rename(columns={'index': 'days from halving'})
	cycle3 = cycle3.rename(columns={'index': 'days from halving'})
	cycle4 = cycle4.rename(columns={'index': 'days from halving'})
	cycle5 = cycle5.rename(columns={'index': 'days from halving'})
 
		# Get the first price value from cycle 5
	first_price_cycle5 = cycle5['price'].iloc[0]

	# cycle1['new_price'] = cycle1['price'].pct_change().add(1).fillna(1).cumprod().mul(first_price_cycle5)
	cycles = [cycle1, cycle2, cycle3, cycle4]

	for cycle in cycles:
		cycle['new_price'] = cycle['price'].pct_change().add(1).fillna(1).cumprod().mul(first_price_cycle5)
	# Add new_date column to cycle3 DataFrame
	cycle3['new_date'] = pd.to_datetime('2024-04-20') + pd.to_timedelta(cycle3['days from halving'], unit='D')
	# cycle1['new_date'] = pd.to_datetime('2024-04-20') + pd.to_timedelta(cycle1['days from halving'], unit='D')
	cycle2['new_date'] = pd.to_datetime('2024-04-20') + pd.to_timedelta(cycle2['days from halving'], unit='D')
	# Add new_date column to cycle4 DataFrame
	cycle4['new_date'] = pd.to_datetime('2024-04-20') + pd.to_timedelta(cycle4['days from halving'], unit='D')
	
	# Create a new figure
	fig = go.Figure()

	# Calculate the average of new_price for cycle3 and cycle4
	average_price = pd.concat([cycle3['new_price'], cycle2['new_price'], cycle4['new_price']], axis=1).mean(axis=1)

	# Convert the average_price Series into a DataFrame
	average_price = average_price.to_frame()

	# Rename the 0 column to 'new_price'
	average_price.columns = ['new_price']

	# Add a new column called 'new_date' and use the 'new_date' column from cycle4
	average_price['new_date'] = cycle4['new_date']

	# Add traces for new_price for cycle3 and cycle4
	fig.add_trace(go.Scatter(x=cycle2.new_date, y=cycle2['new_price'], mode='lines', name='2012-2016 Cycle'))
	fig.add_trace(go.Scatter(x=cycle3.new_date, y=cycle3['new_price'], mode='lines', name='2016-2020 Cycle'))
	fig.add_trace(go.Scatter(x=cycle4.new_date, y=cycle4['new_price'], mode='lines', name='2020-2024 Cycle'))
	fig.add_trace(go.Scatter(x=cycle5.date, y=cycle5['price'], mode='lines', name='2024-2028 Cycle', line=dict(color='black')))

	# Set labels and title
	fig.update_layout(
		title='Cycle Comparison Tracker',
		xaxis_title='Index',
		yaxis_title='New Price',
		width=1200,  # Set the width of the plot
		height=700  # Set the height of the plot
	)
	fig.update_layout(template='plotly_white')
	# Set y-axis to logarithmic scale and format ticks
	fig.update_yaxes(type='log', tickformat='s', tickformatstops=[dict(dtickrange=[None, None], value=",.2s")])
	cycle_comp = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	
	# Cycle Comparison 2
	# Get the most recent 'date' from cycle5
	cycle5['date'] = pd.to_datetime(cycle5['date'])
	recent_date = cycle5.iloc[-1]['date']

	# Filter 'new_date' and 'new_price' for the most recent date
	cycle2_today = cycle2[cycle2['new_date'] == pd.Timestamp(recent_date)][['new_date', 'new_price']]
	cycle3_today = cycle3[cycle3['new_date'] == pd.Timestamp(recent_date)][['new_date', 'new_price']]
	cycle4_today = cycle4[cycle4['new_date'] == pd.Timestamp(recent_date)][['new_date', 'new_price']]
	cycle5_today = cycle5[cycle5['date'] == pd.Timestamp(recent_date)][['date', 'price']]

	# Concatenate the dataframes and rename the 'new_price' or 'price' column
	cycle2_today.rename(columns={'new_date': 'date', 'new_price': 'cycle2_price'}, inplace=True)
	cycle3_today.rename(columns={'new_date': 'date', 'new_price': 'cycle3_price'}, inplace=True)
	cycle4_today.rename(columns={'new_date': 'date', 'new_price': 'cycle4_price'}, inplace=True)
	cycle5_today.rename(columns={'price': 'cycle5_price'}, inplace=True)

	# Merge the dataframes on 'date'
	compare_df = pd.merge(cycle2_today, cycle3_today, on='date', how='outer')
	compare_df = pd.merge(compare_df, cycle4_today, on='date', how='outer')
	compare_df = pd.merge(compare_df, cycle5_today, on='date', how='outer')
 


	# Create a 2x2 grid of subplots of type 'indicator'
	fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("2012 Scaled", "2016 Scaled", "2020 Scaled", "Current"), specs=[[{'type': 'indicator'}, {'type': 'indicator'}], [{'type': 'indicator'}, {'type': 'indicator'}]])

	# Calculate the delta
	delta = compare_df['cycle5_price'].iloc[-1]

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		value = compare_df['cycle2_price'].iloc[-1],
		delta = {'reference': delta}),
		row=1, col=1)

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		value = compare_df['cycle3_price'].iloc[-1],
		delta = {'reference': delta}),
		row=1, col=2)

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		value = compare_df['cycle4_price'].iloc[-1],
		delta = {'reference': delta}),
		row=2, col=1)

	fig.add_trace(go.Indicator(
		mode = "number+delta",
		value = compare_df['cycle5_price'].iloc[-1],
		delta = {'reference': delta}),
		row=2, col=2)
	fig.update_layout(template='plotly_white')
	fig.update_layout(height=500, width=1000, title_text="Halving Comparison 2")
	cycle_comp2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
 
 
    # MSTR
    
	mstr_df = yf.Ticker('MSTR').history(period='7y',interval='1d',actions=False).reset_index()
	mstr_df = mstr_df.rename(columns={"Date": "date"})
	mstr_df['date'] = mstr_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
	mstr_df['price'] = mstr_df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
	mstr_df = mstr_df.drop(columns=['Volume', 'Open', 'High', 'Low', 'Close'])
	# Convert 'date' columns to datetime
	df['date'] = pd.to_datetime(df['date'])
	mstr_df['date'] = pd.to_datetime(mstr_df['date'])


	# Merge the dataframes on the 'date' column
	combined_df = pd.merge(df, mstr_df, on='date', how='outer')
	combined_df = combined_df.rename(columns={"price_x": "bitcoin", "price_y": "mstr"})
	# Drop rows with any NaN values
	combined_df = combined_df.dropna()
	# Filter the combined dataframe to only include data after August 2020
	combined_df = combined_df[combined_df['date'] > '2020-08-01']
	# Create a new column 'mstr_to_bitcoin' by dividing 'mstr' by 'bitcoin'
	combined_df['mstr_to_bitcoin'] = combined_df['mstr']*1000 / combined_df['bitcoin']
	# Display the combined dataframe
	# Calculate moving averages
	combined_df['mstr_to_bitcoin_7d'] = combined_df['mstr_to_bitcoin'].rolling(window=7).mean()
	combined_df['mstr_to_bitcoin_20d'] = combined_df['mstr_to_bitcoin'].rolling(window=20).mean()
	combined_df['mstr_to_bitcoin_50d'] = combined_df['mstr_to_bitcoin'].rolling(window=50).mean()
	combined_df['mstr_to_bitcoin_100d'] = combined_df['mstr_to_bitcoin'].rolling(window=100).mean()
	# Keep only the specified columns
	combined_df = combined_df[['date', 'bitcoin', 'mstr','mstr_to_bitcoin', 'mstr_to_bitcoin_20d']]
	combined_df['mstr_to_bitcoin_diff'] = combined_df['mstr_to_bitcoin'] - combined_df['mstr_to_bitcoin_20d']
	# Calculate the standard deviation of 'mstr_to_bitcoin_diff'
	std_dev = combined_df['mstr_to_bitcoin_diff'].std()

	# Define the bins based on standard deviation
	bins = [-2*std_dev, -std_dev, 0, std_dev, 2*std_dev]

	# Create labels for the bins
	group_names = ['-2 STD', '-1 STD', '+1 STD', '+2 STD']
	combined_df = combined_df.dropna()
	# Create a new column 'std_dev_bucket' with the bin labels
	combined_df['std_dev_bucket'] = pd.cut(combined_df['mstr_to_bitcoin_diff'], bins=bins, labels=group_names)
	fig = px.scatter(combined_df, x="date", y="mstr_to_bitcoin", color="std_dev_bucket", 
					color_discrete_sequence=["blue", "green", "orange", "red"],
					title="MSTR to Bitcoin Ratio with Standard Deviation Buckets")
	fig.add_scatter(x=combined_df['date'], y=combined_df['mstr_to_bitcoin_20d'], mode='lines', name='20 Day MA')
	fig.update_traces(marker=dict(color='black'), selector=dict(name='20 Day MA'))
	fig.update_layout(width=1000, height=700)
	fig.update_layout(template='plotly_white')

	# Add a horizontal gauge bar
	fig.add_trace(go.Indicator(
		mode="gauge+number",
		value=combined_df['mstr_to_bitcoin_diff'].iloc[-1],
		domain={'x': [0.1, 0.9], 'y': [0.85, 1]},
		title={'text': "MSTR to Bitcoin Diff"},
		gauge={
			'shape': "bullet",
			'axis': {'range': [-2*std_dev, 2*std_dev]},
			'bar': {'color': "black"},
			'steps': [
				{'range': [-2*std_dev, -std_dev], 'color': "red"},
				{'range': [-std_dev, 0], 'color': "orange"},
				{'range': [0, std_dev], 'color': "blue"},
				{'range': [std_dev, 2*std_dev], 'color': "green"}
			],
			'threshold': {
				'line': {'color': "black", 'width': 4},
				'thickness': 0.3,
				'value': combined_df['mstr_to_bitcoin_diff'].iloc[-1]
			}
		}
	))

	fig.update_traces(
		domain={'x': [0.4, 0.9], 'y': [0.85, 0.95]},
		selector=dict(type='indicator')
	)
	fig.show()
	MSTR = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template('bar.html',MSTR=MSTR,cycle_comp2=cycle_comp2,cycle_comp=cycle_comp,Buyzonesbar=Buyzonesbar,cloud=cloud,Rainbow=Rainbow,BRainbow=BRainbow,Movingaverages2=Movingaverages2,corr2=corr2,corr1=corr1,YTD=YTD, Buyzones=Buyzones, Movingaverages=Movingaverages,Indicators=Indicators)

	
if __name__ == '__main__':
	app.run()

