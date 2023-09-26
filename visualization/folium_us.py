# load the GeoJSON data for US states
us_states = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'

# create the map
map_airport = folium.Map(location=[us_airport_loc['LATITUDE'].mean(
), us_airport_loc['LONGITUDE'].mean()], zoom_start=4)

# add airport markers to the map
for index, row in us_airport_loc.iterrows():
    folium.CircleMarker(location=[
                        row['LATITUDE'], row['LONGITUDE']], radius=3, fill=True).add_to(map_airport)


# define the style function
def style_function(feature):
    return {
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.07,
        'dashArray': '5, 5'
    }


# create the GeoJson layer with the style function
folium.GeoJson(f'{us_states}/us-states.json',
               name='US States',
               style_function=style_function).add_to(map_airport)

# display the map
map_airport
