import streamlit as st
import googlemaps
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

# Page config
st.set_page_config(page_title="Carpool Meetup Optimizer", page_icon="🚗", layout="wide")

# Initialize session state for results
if 'result' not in st.session_state:
    st.session_state.result = None
if 'origin_coords' not in st.session_state:
    st.session_state.origin_coords = None
if 'origin_formatted' not in st.session_state:
    st.session_state.origin_formatted = None
if 'dest_coord' not in st.session_state:
    st.session_state.dest_coord = None
if 'dest_formatted' not in st.session_state:
    st.session_state.dest_formatted = None
if 'num_origins' not in st.session_state:
    st.session_state.num_origins = None
if 'togetherness' not in st.session_state:
    st.session_state.togetherness = None

def geocode_address(gmaps, address):
    """Convert address to lat/lon coordinates."""
    try:
        result = gmaps.geocode(address)
        if result:
            loc = result[0]['geometry']['location']
            return (loc['lat'], loc['lng']), result[0]['formatted_address']
        return None, None
    except Exception as e:
        st.error(f"Geocoding error for '{address}': {e}")
        return None, None

def generate_candidate_grid(origins, destination, grid_size=5):
    """Generate candidate meetup points between origins centroid and destination."""
    origin_lats = [o[0] for o in origins]
    origin_lngs = [o[1] for o in origins]
    centroid_lat = np.mean(origin_lats)
    centroid_lng = np.mean(origin_lngs)
    dest_lat, dest_lng = destination
    lat_min = min(centroid_lat, dest_lat, min(origin_lats))
    lat_max = max(centroid_lat, dest_lat, max(origin_lats))
    lng_min = min(centroid_lng, dest_lng, min(origin_lngs))
    lng_max = max(centroid_lng, dest_lng, max(origin_lngs))
    padding = 0.1
    lat_range = lat_max - lat_min
    lng_range = lng_max - lng_min
    lat_min -= lat_range * padding
    lat_max += lat_range * padding
    lng_min -= lng_range * padding
    lng_max += lng_range * padding
    lats = np.linspace(lat_min, lat_max, grid_size)
    lngs = np.linspace(lng_min, lng_max, grid_size)
    candidates = [(lat, lng) for lat in lats for lng in lngs]
    return candidates

def get_drive_times(gmaps, origins, destinations, departure_time=None):
    """Get drive times matrix using Distance Matrix API."""
    try:
        result = gmaps.distance_matrix(
            origins=origins,
            destinations=destinations,
            mode="driving",
            departure_time=departure_time,
            traffic_model="best_guess"
        )
        times = []
        for row in result['rows']:
            row_times = []
            for element in row['elements']:
                if element['status'] == 'OK':
                    if 'duration_in_traffic' in element:
                        row_times.append(element['duration_in_traffic']['value'])
                    else:
                        row_times.append(element['duration']['value'])
                else:
                    row_times.append(float('inf'))
            times.append(row_times)
        return times
    except Exception as e:
        st.error(f"Distance Matrix API error: {e}")
        return None

def calculate_meetup_cost(origin_times, destination_time, num_people, togetherness=0):
    """
    Calculate total cost with togetherness preference.
    
    togetherness = 0: minimize total person-time (meet near destination)
    togetherness = 100: prioritize driving together (meet near origins)
    
    The multiplier on group leg ranges from n (at 0%) to 1 (at 100%).
    At 100%, the group drive only "counts once" regardless of passengers.
    """
    togetherness_factor = togetherness / 100.0
    group_multiplier = num_people - togetherness_factor * (num_people - 1)
    return sum(origin_times) + (group_multiplier * destination_time)

def find_optimal_meetup(gmaps, origin_coords, dest_coord, departure_time, grid_size=5, refine_iterations=2, togetherness=50):
    """Find optimal meetup point using grid search with refinement."""
    num_people = len(origin_coords)
    best_candidate = None
    best_cost = float('inf')
    best_origin_times = None
    best_dest_time = None
    candidates = generate_candidate_grid(origin_coords, dest_coord, grid_size)
    
    for iteration in range(refine_iterations):
        progress_text = f"Iteration {iteration + 1}/{refine_iterations}: Testing {len(candidates)} candidates..."
        progress_bar = st.progress(0, text=progress_text)
        batch_size = 10
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]
            origin_times = get_drive_times(gmaps, origin_coords, batch, departure_time)
            if origin_times is None:
                continue
            dest_times = get_drive_times(gmaps, batch, [dest_coord], departure_time)
            if dest_times is None:
                continue
            for i, candidate in enumerate(batch):
                times_to_candidate = [origin_times[j][i] for j in range(num_people)]
                time_to_dest = dest_times[i][0]
                if float('inf') in times_to_candidate or time_to_dest == float('inf'):
                    continue
                cost = calculate_meetup_cost(times_to_candidate, time_to_dest, num_people, togetherness)
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = candidate
                    best_origin_times = times_to_candidate
                    best_dest_time = time_to_dest
            progress = min((batch_start + batch_size) / len(candidates), 1.0)
            progress_bar.progress(progress, text=progress_text)
        progress_bar.empty()
        if best_candidate and iteration < refine_iterations - 1:
            lat, lng = best_candidate
            lat_range = 0.02 / (iteration + 1)
            lng_range = 0.02 / (iteration + 1)
            lats = np.linspace(lat - lat_range, lat + lat_range, grid_size)
            lngs = np.linspace(lng - lng_range, lng + lng_range, grid_size)
            candidates = [(la, ln) for la in lats for ln in lngs]
    
    if best_candidate:
        return {
            'location': best_candidate,
            'total_cost': best_cost,
            'origin_times': best_origin_times,
            'destination_time': best_dest_time
        }
    return None

def format_duration(seconds):
    """Convert seconds to human-readable duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

def create_map(origin_coords, origin_addresses, dest_coord, dest_address, meetup_point):
    """Create a folium map showing all points."""
    all_lats = [o[0] for o in origin_coords] + [dest_coord[0], meetup_point[0]]
    all_lngs = [o[1] for o in origin_coords] + [dest_coord[1], meetup_point[1]]
    center_lat = np.mean(all_lats)
    center_lng = np.mean(all_lngs)
    m = folium.Map(location=[center_lat, center_lng], zoom_start=10)
    colors = ['blue', 'purple', 'orange', 'darkblue']
    for i, (coord, addr) in enumerate(zip(origin_coords, origin_addresses)):
        folium.Marker(
            location=coord,
            popup=f"Origin {i+1}: {addr}",
            icon=folium.Icon(color=colors[i % len(colors)], icon='home')
        ).add_to(m)
    folium.Marker(
        location=dest_coord,
        popup=f"Destination: {dest_address}",
        icon=folium.Icon(color='red', icon='flag')
    ).add_to(m)
    folium.Marker(
        location=meetup_point,
        popup=f"Optimal Meetup Point",
        icon=folium.Icon(color='green', icon='star')
    ).add_to(m)
    folium.Circle(
        location=meetup_point,
        radius=500,
        color='green',
        fill=True,
        fillOpacity=0.3
    ).add_to(m)
    return m

# Main app
st.title("🚗 Carpool Meetup Optimizer")
st.markdown("""
Find the optimal meetup point for your carpool that minimizes total drive time 
for everyone, accounting for real-time traffic conditions.
""")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Google Maps API Key", type="password", 
                            help="Enter your Google Maps API key")
    st.markdown("---")
    st.markdown("""
    **How to get an API key:**
    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a new project
    3. Enable these APIs:
       - Geocoding API
       - Distance Matrix API
    4. Create credentials (API key)
    5. Set up billing (you get $200/month free)
    """)
    st.markdown("---")
    grid_size = st.slider("Grid resolution", 3, 7, 5, 
                          help="Higher = more accurate but more API calls")
    refine_iterations = st.slider("Refinement iterations", 1, 3, 2,
                                  help="More iterations = better precision")
    st.markdown("---")
    togetherness = st.slider("🚗 Group drive preference", 0, 100, 50,
                             help="0% = minimize total time (meet near destination). 100% = maximize time together (meet near origins).")
    st.caption("Low = efficient but separate. High = more carpool time together.")

# Main form
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Starting Locations")
    num_origins = st.selectbox("Number of people/cars", [2, 3, 4], index=1)
    origins = []
    for i in range(num_origins):
        addr = st.text_input(f"Person {i+1} starting address", 
                            key=f"origin_{i}",
                            placeholder="123 Main St, City, State")
        if addr:
            origins.append(addr)

with col2:
    st.subheader("🏁 Destination")
    destination = st.text_input("Final destination address",
                               placeholder="456 Mountain Rd, City, State")
    st.subheader("🕐 Departure Time")
    use_current_time = st.checkbox("Leave now", value=True)
    if not use_current_time:
        date = st.date_input("Date", datetime.now())
        time = st.time_input("Time", datetime.now().time())
        departure_datetime = datetime.combine(date, time)
    else:
        departure_datetime = datetime.now() + timedelta(minutes=5)

# Calculate button
if st.button("🔍 Find Optimal Meetup Point", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please enter your Google Maps API key in the sidebar.")
    elif len(origins) < 2:
        st.error("Please enter at least 2 starting addresses.")
    elif not destination:
        st.error("Please enter a destination address.")
    else:
        gmaps = googlemaps.Client(key=api_key)
        with st.spinner("Geocoding addresses..."):
            origin_coords = []
            origin_formatted = []
            for addr in origins:
                coord, formatted = geocode_address(gmaps, addr)
                if coord:
                    origin_coords.append(coord)
                    origin_formatted.append(formatted)
                else:
                    st.error(f"Could not geocode: {addr}")
            dest_coord, dest_formatted = geocode_address(gmaps, destination)
        
        if len(origin_coords) == len(origins) and dest_coord:
            st.info(f"Searching for optimal meetup point (departure: {departure_datetime.strftime('%Y-%m-%d %H:%M')})...")
            result = find_optimal_meetup(
                gmaps, origin_coords, dest_coord, 
                departure_datetime, grid_size, refine_iterations, togetherness
            )
            
            if result:
                # Store results in session state
                st.session_state.result = result
                st.session_state.origin_coords = origin_coords
                st.session_state.origin_formatted = origin_formatted
                st.session_state.dest_coord = dest_coord
                st.session_state.dest_formatted = dest_formatted
                st.session_state.num_origins = len(origins)
                st.session_state.togetherness = togetherness
            else:
                st.error("Could not find an optimal meetup point. Try adjusting the addresses.")
                st.session_state.result = None

# Display results from session state (persists across reruns)
if st.session_state.result:
    result = st.session_state.result
    origin_coords = st.session_state.origin_coords
    origin_formatted = st.session_state.origin_formatted
    dest_coord = st.session_state.dest_coord
    dest_formatted = st.session_state.dest_formatted
    num_origins = st.session_state.num_origins
    togetherness = st.session_state.togetherness
    
    st.success("✅ Found optimal meetup point!")
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.subheader("📍 Optimal Meetup Location")
        lat, lng = result['location']
        st.markdown(f"**Coordinates:** `{lat:.6f}, {lng:.6f}`")
        maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
        st.markdown(f"[🗺️ Open in Google Maps]({maps_url})")
        m = create_map(origin_coords, origin_formatted, 
                      dest_coord, dest_formatted, result['location'])
        st_folium(m, width=600, height=400, returned_objects=[])
    
    with res_col2:
        st.subheader("⏱️ Drive Times")
        st.markdown("**To meetup point:**")
        for i, t in enumerate(result['origin_times']):
            st.markdown(f"- Person {i+1}: {format_duration(t)}")
        st.markdown(f"**Meetup → Destination:** {format_duration(result['destination_time'])}")
        st.markdown("---")
        st.markdown("**📊 Trip Breakdown**")
        individual_total = sum(result['origin_times'])
        group_time = result['destination_time']
        avg_individual = individual_total / num_origins
        total_trip_per_person = avg_individual + group_time
        pct_together = (group_time / total_trip_per_person * 100) if total_trip_per_person > 0 else 0
        st.markdown(f"- Avg drive to meetup: {format_duration(avg_individual)}")
        st.markdown(f"- Group drive together: {format_duration(group_time)}")
        st.markdown(f"- **Time together: {pct_together:.0f}%** of each person's trip")
        st.markdown("---")
        st.markdown(f"*Group preference setting: {togetherness}%*")
    
    # Clear results button
    if st.button("🔄 Clear Results"):
        st.session_state.result = None
        st.rerun()

st.markdown("---")
st.markdown("""
<small>
💡 **Tips:** 
- The green marker shows the optimal meetup area - find a parking lot or business nearby
- Times include current traffic conditions when available
- Increase grid resolution for more accuracy (uses more API calls)
</small>
""", unsafe_allow_html=True)
