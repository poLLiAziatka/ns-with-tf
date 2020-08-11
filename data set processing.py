f0 = open('Patio_Lawn_and_Garden_5.json', 'r')
f1 = open('dataset', 'a')
for line in f0:
    need_data = [line[line.find('reviewText') + 14: line.find('overall') - 4]]
    f1.write(need_data[0] + ', "patio, lawn and garden"' + '\n')
f0.close()
f1.close()