import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pylab import rc, rcParams

def readData(base_path, file,choice):
	df= pd.read_csv(base_path+file)
	df.drop(["Lat","Long"], axis=1, inplace=True)
	df=df.melt(id_vars=["Country/Region","Province/State"], 
	        var_name="Date", 
	        value_name=choice)
	df['Date'] = pd.to_datetime(df['Date'])
	df['Date']=df['Date'].apply(lambda x:x.date())
	return df

def casePlot(df, area, choice, k):
    df = df.groupby([area])[choice].max().reset_index().sort_values(choice,ascending=False).nlargest(k,choice)
    rcParams['figure.figsize'] = 20, 10 
    sns.barplot(x=choice, y=area,order=df[area], data=df)
    plt.yticks(rotation=50)
    plt.ylabel(area,fontsize=15)
    plt.xlabel('Number of '+choice,fontsize=15)
    plt.show()

