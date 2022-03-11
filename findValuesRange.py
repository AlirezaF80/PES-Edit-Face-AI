from extractNamesFromPics import extract_csv


def save_json(data, filename):  # using json library
    import json
    with open(filename, 'w') as f:
        json.dump(data, f)


csv_file = extract_csv('data-cleaned.csv')
appearances = {line[0]: [int(i) for i in line[4:]] for line in csv_file[1:]}
appearances['Id'] = csv_file[0][4:]
model_save_path = 'modelNew.h5'

if __name__ == '__main__':
    min_max_values = {}
    for i in range(len(appearances['Id'])):
        all_values = [appearances[key][i] for key in appearances if key != 'Id']
        min_max_values[appearances['Id'][i]] = (min(all_values), max(all_values))
    save_json(min_max_values, 'values_range.json')
