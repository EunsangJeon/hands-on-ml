import matplotlib.pyplot as pyplot
import numpy
import pandas
from sklearn.linear_model import LinearRegression

# load data
oecd_bli_2015 = pandas.read_csv("../dataset/oecd_bli_2015.csv", thousands=",")
gdp_per_capita_2015 = pandas.read_csv("../dataset/gdp_per_capita.csv", thousands=",",
                                      delimiter="\t", encoding="latin1", na_values="n/a")


# prepare data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pandas.merge(left=oecd_bli, right=gdp_per_capita,
                                      left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    return full_country_stats[["GDP per capita", "Life satisfaction"]]

country_stats = prepare_country_stats(oecd_bli_2015, gdp_per_capita_2015)
x = numpy.c_[country_stats["GDP per capita"]]
y = numpy.c_[country_stats["Life satisfaction"]]

# visualize data
country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
pyplot.show()

# chose linear model
model = LinearRegression()

# train model
model.fit(x, y)

# predict
x_new = [[25000], [35000]]
print(model.predict(x_new))
