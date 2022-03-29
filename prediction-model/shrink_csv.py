import dask.dataframe as dd

labels = {
    'time': 'Trip Start Timestamp',
    'lat_a': 'Pickup Centroid Latitude',
    'long_a': 'Pickup Centroid Longitude',
    'lat_b': 'Dropoff Centroid Latitude',
    'long_b': 'Dropoff Centroid Longitude'
}

# df = dd.read_csv('data/trips.csv', usecols=labels.values(), dtype={'Dropoff Centroid Latitude': 'object'})
df = dd.read_csv('data/trips_shrunk.csv/*', dtype={'Dropoff Centroid Latitude': 'object'})
df.to_csv('data/trips_shrunk_single.csv', index=False, single_file=True)
