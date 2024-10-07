import matplotlib.pyplot
import matplotlib.pyplot
import matplotlib.pyplot


def sarimax(path):
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import matplotlib
    matplotlib.use('Agg')
    import os

    # Create a folder to save the plots
    folder_path = os.path.join(os.path.dirname(path), 'plots')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Load multivariate sales data from CSV file
    sales_data = pd.read_csv(path, parse_dates=['Dates'], index_col='Dates')

    # List of product names
    products = sales_data.columns

    # Iterate over each product and train SARIMAX model
    for product in products:
        # Train SARIMAX model
        model = SARIMAX(sales_data[product], order=(5,1,0), seasonal_order=(1,1,1,12))  # SARIMAX(p,d,q)(P,D,Q,s)
        model_fit = model.fit()

        # Forecast future sales
        n_forecast_steps = 12  # Forecasting sales for the next 12 months
        forecast = model_fit.forecast(steps=n_forecast_steps)

        # Print the forecasted sales
        print(f'Product: {product}')
        print(f'Forecasted Sales: {forecast}\n')

        # Visualize past sales and forecasted sales for each product separately
        matplotlib.pyplot.figure(figsize=(12, 6))
        matplotlib.pyplot.plot(sales_data.index, sales_data[product], label='Past Sales', color='#041723')
        matplotlib.pyplot.plot(pd.date_range(start=sales_data.index[-1], periods=n_forecast_steps , freq='ME')[0:], forecast, label='Forecasted Sales', linestyle='-', color='#e37439')
        matplotlib.pyplot.xlabel('Date')
        matplotlib.pyplot.ylabel('Sales')
        matplotlib.pyplot.title(f'SARIMAX Forecast of Future Sales for Product: {product}')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.grid(True)

        plot_path = os.path.join(folder_path, f'{product}_plot.png')
        matplotlib.pyplot.savefig(plot_path)  # Increase dpi for a larger image

        # Close the plot to release memory
        matplotlib.pyplot.close()

sarimax('D:\supply_chain_optimization\media\SalesData1.csv')