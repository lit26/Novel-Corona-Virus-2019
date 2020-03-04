import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rc, rcParams

def casePlot(df, area, choice):
    df = df.groupby([area])[choice].max().reset_index().\
                        sort_values(choice,ascending=False).nlargest(15,choice)
    sns.barplot(x=choice, y=area,order=df[area], \
                 data=df)
    plt.yticks(rotation=50)
    plt.ylabel(area,fontsize=15)
    plt.xlabel('Number of '+choice,fontsize=15)
    plt.show()