import pandas as pd

table_kucun = pd.read_excel('/Users/haha/Desktop/table (2).xls', name=['kuwei', 'value'])
table_cons = pd.read_excel('/Users/haha/Desktop/table (3).xls', name=['kuwei', 'value'])

table_kucun['location'] = table_kucun['location'].map(str.strip)
table_cons['location'] = table_cons['location'].map(str.strip)

quant_location = table_kucun['location']
quant_cost = table_kucun['cost']

consigt_location = table_cons['location']
cosigt_cost = table_cons['cost']

iter_size = table_cons.shape[0]
location = []
cost = []
con_cost = []


def iter_table_consigt(location_name, cost):
    for i in range(iter_size - 1):
        if consigt_location[i] == location_name and round(cost, 2) != round(cosigt_cost[i], 2):
            location.append(location_name)
            quant_cost.append(round(cost, 2))
            con_cost.append( round(cosigt_cost[i], 2))
            print(location_name, round(cost, 2),  round(cosigt_cost[i], 2))
        elif consigt_location[i] == location_name and round(cost, 2) == round(cosigt_cost[i], 2):
            break


t = table_kucun.shape[0]
for j in range(t - 1):
    ln = quant_location[j]
    co = quant_cost[j]
    iter_table_consigt(ln, co)


d = {'location': location, 'quant_cost': quant_cost, 'con_cost': con_cost}
df = pd.DataFrame(data=d)