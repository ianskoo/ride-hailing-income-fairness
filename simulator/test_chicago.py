from chicago_data import Chicago
from datetime import date, time
from tqdm import tqdm


def test_coordinates_within_grid(ch: Chicago):

    print("Testing whether converted coordinate are within grid...")
    lat_a = ch.df.loc[:, ch.labels['lat_a']].values
    long_a = ch.df.loc[:, ch.labels['long_a']].values

    for lat, long in tqdm(zip(lat_a, long_a)):
        y_a, x_a = ch.coord_to_grid(lat, long)
        assert(x_a < ch.n and y_a < ch.m)

    lat_b = ch.df.loc[:, ch.labels['lat_b']].values
    long_b = ch.df.loc[:, ch.labels['long_b']].values

    for lat, long in tqdm(zip(lat_b, long_b)):
        y_b, x_b = ch.coord_to_grid(lat, long)
        assert(x_b < ch.n and y_b < ch.m)


chic = Chicago('data/test_trips.csv',
               "2020-1-6",
               "2020-1-13",
               (time(8), time(10)),
               (time(17), time(21)))

# print(f"Chicago city size (m): {chic.city_size}")
# print(f"Simulation grid size: {chic.n} x {chic.m}")
print(chic.df)

print(chic.create_chicago_taxi_home_coords())
# test_coordinates_within_grid(chic)

# for rate in range(1, 5):
#     print(chic.get_request_rate(rate))

# for sim_time in range(0, 270, 90):
#     print(sim_time, chic.sim_time_to_datetime(sim_time))
#     print(chic.choose_one_request_coord(sim_time), "\n")
