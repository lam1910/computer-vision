from google_images_download import google_images_download   #importing the library
import json
import pandas as pd

response = google_images_download.googleimagesdownload()

if __name__ == "__main__":
    filepath = 'data/raw/100VietnameseCelebName.txt'
    df = pd.DataFrame(columns=['Name', 'Pic'])

    name_list = pd.read_csv(filepath, header=None)
    name_list.columns = ['name']
    # name_list.apply(lambda x: x.strip())

    for _, row in name_list.iterrows():
        name = row.values[0]
        paths, err = response.download({"keywords": name, "limit": 10, 'format': 'png'})
        a = json.dumps(paths)
        d = json.loads(a)
        temp = []
        for i in range(0, len(d[name])):
            temp.append(name)
        data = {'Name': temp, 'Pic': d[name]}
        df = df.append(data, ignore_index=True)

    df.to_csv('celeb.csv')
