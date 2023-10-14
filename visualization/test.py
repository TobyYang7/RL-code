global ax
global fig
global map
fig, ax = plt.subplots(figsize=(30, 30))
ax.set_title('USA Map')


def plot_map(time_slot):
    map = Basemap(projection='merc', llcrnrlon=-160, llcrnrlat=15,
                  urcrnrlon=-60, urcrnrlat=65, resolution='l')
    # map = Basemap(width=11000000, height=5500000,
    #               rsphere=(6378137.00, 6356752.3142),
    #               resolution='l', area_thresh=1000., projection='lcc',
    #               lat_1=45., lat_2=55, lat_0=45, lon_0=-110.)

    # plots US map
    map.drawcoastlines(linewidth=0.5)
    map.drawstates(linewidth=0.5)
    map.drawcountries(linewidth=1.5)
    map.fillcontinents(color='lightgray', lake_color='lightblue')
    map.drawmapboundary(fill_color='lightblue')

    # map.drawparallels(np.arange(15, 70., 20.), labels=[
    #                   1, 0, 0, 0], color='black', linewidth=0.5)
    # map.drawmeridians(np.arange(-210., -60., 20.),
    #                   labels=[0, 0, 0, 1], color='black', linewidth=0.5)

    # todo: adjust size, time_slot
    plot_airports(map, s=50, time_slot=time_slot)
    # plot_OD_flow(map)


def plot_airports(map, s, time_slot):
    time_slot = time_slot

    arr_delay = df_arr.loc[time_slot, :].fillna(0)
    dep_delay = df_dep.loc[time_slot, :].fillna(0)

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(dep_delay.min(), dep_delay.max())

    for airport in us_airport_loc['Name']:
        lat, lon = get_airport_location(airport)
        x, y = map(lon, lat)
        arr = arr_delay[airport]
        dep = dep_delay[airport]

        size = s * (arr-arr_delay.min()) / (arr_delay.max()-arr_delay.min())
        color = cmap(norm(dep))

        map.scatter(x, y, s=size*size, color=color, marker='o', alpha=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Departure Delay(min)', shrink=0.5)


def plot_flight_route(dep, arr, map, linewidth=2, color='b', alpha=1.0):
    dep_lat, dep_lon = get_airport_location(dep)
    arr_lat, arr_lon = get_airport_location(arr)
    map.drawgreatcircle(dep_lon, dep_lat, arr_lon,
                        arr_lat, linewidth=linewidth, color=color, alpha=alpha)


def plot_OD_flow(map):
    # create a new dataframe that sums the flow for each origin-destination pair
    flow_df = pd.DataFrame({'Origin': us_airport_loc['Name'].values[OD.nonzero()[0]],
                            'Destination': us_airport_loc['Name'].values[OD.nonzero()[1]],
                            'Flow': OD[OD.nonzero()]})
    flow_df = flow_df.groupby(['Origin', 'Destination']).sum().reset_index()

    # sort the dataframe by flow in descending order
    flow_df = flow_df.sort_values(by='Flow', ascending=False)

    # define a colormap and normalize flow values
    cmap = plt.get_cmap('Blues')
    norm = plt.Normalize(flow_df['Flow'].min(), flow_df['Flow'].max())

    # plot the top 100 routes with colors based on flow
    for i in range(70):
        dep = flow_df.iloc[i]['Origin']
        arr = flow_df.iloc[i]['Destination']
        flow = flow_df.iloc[i]['Flow']

        color = cmap(norm(flow))
        linewidth = 0.1 + 1.8 * \
            (flow - flow_df['Flow'].min()) / \
            (flow_df['Flow'].max() - flow_df['Flow'].min())
        alpha = (flow - flow_df['Flow'].min()) / \
            (flow_df['Flow'].max() - flow_df['Flow'].min())

        plot_flight_route(dep, arr, map, linewidth=linewidth * linewidth,
                          color=color, alpha=alpha * alpha)

    # Add a colorbar to the map to show the flow values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='OD Value', shrink=0.2, location='left')


plot_map(10)
