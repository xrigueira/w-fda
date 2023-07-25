(This has not been done yet. I am just typing it to have it ready just in case.)

The two following anomalies have been removed from anomalies.csv:

901;05-03-2013 02:00:00;11-03-2013 00:00:00;turbidity, water_level
901;14-10-2017 18:00:00;15-10-2017 18:00:00;water_level

This is because when working on the functional problem and station 901,
which is the one thought most fitted for this problem, the variables
turbidity and water flow have been removed because they were affecting
the results negative. This is due to both of them not being key in
the vast majority of anomalies, but when there is a spike in turbidity,
for instance, the functional model pick it up when it shouldn't be an
anomaly as it is not labeled.