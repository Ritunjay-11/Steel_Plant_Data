import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure 'openpyxl' is installed
try:
    import openpyxl
except ImportError:
    st.error("Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl.")
    st.stop()

# Load the provided Excel file for water quality
water_quality_file_path = 'Water_Quality_Dataset_with_Anomalies.xlsx'
water_quality_data = pd.read_excel(water_quality_file_path, engine='openpyxl')

# Function to calculate LSI
def calculate_lsi(ph, tds, temp, calcium_hardness, alkalinity):
    A = np.log10(tds) - 1
    B = (-13.12 * np.log10(temp + 273)) + 34.55
    C = np.log10(calcium_hardness) - 0.4
    D = np.log10(alkalinity)
    pHs = (9.3 + A + B) - (C + D)
    LSI = ph - pHs
    return LSI

def categorize_lsi(lsi):
    if lsi < -2:
        return 'Highly Corrosive'
    elif -2 <= lsi < -0.5:
        return 'Moderately Corrosive'
    elif -0.5 <= lsi < 0:
        return 'Slightly Corrosive to Balanced'
    elif 0 <= lsi < 0.5:
        return 'Slightly Scaling'
    elif 0.5 <= lsi < 2:
        return 'Moderately Scaling'
    else:
        return 'Highly Scaling'

def categorize_ph(ph):
    if ph < 6.5:
        return 'Acidic'
    elif 6.5 <= ph <= 9:
        return 'Normal'
    else:
        return 'Alkaline'

# Add a column for pH categories
water_quality_data['pH Category'] = water_quality_data['pH'].apply(categorize_ph)

water_quality_data['LSI'] = water_quality_data.apply(
    lambda row: calculate_lsi(row['pH'], row['TDS'], row['Temperature'], row['Hardness'], row['Alkalinity']), axis=1
)
water_quality_data['LSI Category'] = water_quality_data['LSI'].apply(categorize_lsi)

# Set up the Streamlit app
st.title('Water Consumption and Quality in a Steel Plant')

# Sidebar for navigation
st.sidebar.header('Navigation')
nav_option = st.sidebar.selectbox('Choose a section', ['Water Consumption', 'Water Quality'])

if nav_option == 'Water Consumption':
    # Load the provided Excel file for water consumption
    file_path = 'Synthetic_Dataset_Tata_Steel.xlsx'
    data = pd.read_excel(file_path, engine='openpyxl')

    # Clean the data
    data.columns = data.iloc[0]
    data = data[1:]

    # Rename columns to make them unique and easier to understand
    columns = ['Date', 'Amiad_Filter_IR', 'Amiad_Filter_FR', 'Amiad_Filter_Consumption',
               'Scale_Pit_IR', 'Scale_Pit_FR', 'Scale_Pit_Consumption',
               'Cold_Well_IR', 'Cold_Well_FR', 'Cold_Well_Consumption',
               'Cooling_Tower_IR', 'Cooling_Tower_FR', 'Cooling_Tower_Consumption',
               'PPF_Pit_IR', 'PPF_Pit_FR', 'PPF_Pit_Consumption']
    data.columns = columns

    # Convert 'Date' to datetime and extract day of the month
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day'] = data['Date'].dt.day

    # Sidebar for date range selection
    st.sidebar.header('Filter Data')
    start_date = st.sidebar.date_input('Start date', data['Date'].min())
    end_date = st.sidebar.date_input('End date', data['Date'].max())

    # Filter data based on date range
    filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

    # Site selection menu
    site_options = {
        'Amiad Filter': 'Amiad_Filter_Consumption',
        'Scale Pit': 'Scale_Pit_Consumption',
        'Cold Well': 'Cold_Well_Consumption',
        'Cooling Tower': 'Cooling_Tower_Consumption',
        'PPF Pit': 'PPF_Pit_Consumption'
    }
    selected_site = st.sidebar.selectbox('Select Site', list(site_options.keys()))
    col_prefix = site_options[selected_site]

    # Plotting function for bar chart with mean line
    def plot_consumption(df, title, col_prefix):
        fig, ax = plt.subplots(figsize=(10, 6))
        df = df[['Day', col_prefix]].dropna()
        sns.barplot(x='Day', y=col_prefix, data=df, ax=ax, palette='Blues_d')
        mean_value = df[col_prefix].mean()
        ax.axhline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean Consumption: {mean_value:.2f}')
        ax.set_title(title)
        ax.set_xlabel('Day of the Month')
        ax.set_ylabel('Consumption')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Plotting function for trend analysis of selected site
    def plot_trend(df, title, col_prefix):
        fig, ax = plt.subplots(figsize=(10, 6))
        df = df[['Day', col_prefix]].dropna()
        sns.lineplot(x='Day', y=col_prefix, data=df, ax=ax, marker='o', color='blue')
        ax.set_title(title)
        ax.set_xlabel('Day of the Month')
        ax.set_ylabel('Consumption')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Identify anomalies for the selected site
    def detect_anomalies(df, col_prefix):
        df = df[['Date', 'Day', col_prefix]].dropna()
        mean_value = df[col_prefix].astype(float).mean()
        threshold = 38.57  # Specific deviation threshold
        anomalies = df[(df[col_prefix].astype(float) > mean_value + threshold) | (df[col_prefix].astype(float) < mean_value - threshold)]
        
        return anomalies, mean_value, threshold

    # Generate predictions based on causes and predictions for high or low consumption
    def generate_predictions(anomalies, site, col_prefix, mean_value, threshold):
        causes_high = []
        predictions_high = []
        causes_low = []
        predictions_low = []

        if site == 'Amiad Filter':
            causes_high = ["Increased Production Rates", "Frequent Backwashing", "Higher Particulate Load"]
            predictions_high = ["Increased production demands leading to higher water throughput.",
                                "Possible maintenance activities or issues causing more frequent backwashing."]
            causes_low = ["Reduced Production Rates", "Efficient Filtration", "Process Shutdowns"]
            predictions_low = ["Reduced operational activities or process optimization.",
                               "Potential shutdowns or slowdowns in production."]
        elif site == 'Scale Pit':
            causes_high = ["Increased Cooling Water Usage", "More Scale Formation", "Leakage or Overflows"]
            predictions_high = ["Increased cooling requirements or operational inefficiencies.",
                                "Potential equipment issues or leaks causing higher water flow."]
            causes_low = ["Reduced Cooling Water Usage", "Process Optimization", "Maintenance Shutdowns"]
            predictions_low = ["Lower production demands or optimized processes.",
                               "Temporary operational shutdowns or maintenance activities."]
        elif site == 'Cold Well':
            causes_high = ["Increased Evaporation Losses", "Leaks or Overflows", "Higher Ambient Temperatures"]
            predictions_high = ["Higher ambient temperatures or increased production leading to more water loss.",
                                "Possible system leaks or inefficiencies."]
            causes_low = ["Reduced Evaporation Losses", "System Efficiency", "Lower Production Rates"]
            predictions_low = ["Cooler ambient temperatures or lower production levels.",
                               "Improved water management practices."]
        elif site == 'Cooling Tower':
            causes_high = ["Increased Cooling Demand", "System Leaks", "Higher Ambient Temperatures"]
            predictions_high = ["Increased production or higher ambient temperatures leading to higher cooling needs.",
                                "Potential leaks or inefficiencies in the cooling system."]
            causes_low = ["Reduced Cooling Demand", "Improved Cooling Efficiency", "Production Slowdown"]
            predictions_low = ["Lower ambient temperatures or production levels.",
                               "Optimized cooling processes or temporary shutdowns."]
        elif site == 'PPF Pit':
            causes_high = ["Increased Process Water Usage", "Leaks or Overflows", "Maintenance Activities"]
            predictions_high = ["Increased production demands or inefficiencies in water usage.",
                                "Possible system leaks or maintenance activities causing higher water flow."]
            causes_low = ["Reduced Process Water Usage", "Improved Water Efficiency", "Production Slowdown"]
            predictions_low = ["Reduced operational activities or optimized water usage processes.",
                               "Temporary shutdowns or maintenance leading to lower water intake."]

        predictions = []
        for _, row in anomalies.iterrows():
            if row[col_prefix] > mean_value + threshold:
                predictions.append([row['Date'], row[col_prefix], "Unusually High", "<br>".join(causes_high), "<br>".join(predictions_high)])
            elif row[col_prefix] < mean_value - threshold:
                predictions.append([row['Date'], row[col_prefix], "Unusually Low", "<br>".join(causes_low), "<br>".join(predictions_low)])
        
        predictions_df = pd.DataFrame(predictions, columns=["Date", "Consumption", "Unusually High/Low", "Causes", "Predictions"])
        return predictions_df

    # Display selected site data and trend analysis
    st.subheader(f'{selected_site} Consumption')
    plot_consumption(filtered_data, f'{selected_site} Consumption', col_prefix)
    st.markdown("<br>", unsafe_allow_html=True)

    # Display trend analysis for the selected site
    st.subheader(f'{selected_site} Consumption Trend')
    plot_trend(filtered_data, f'{selected_site} Consumption Trend', col_prefix)
    st.markdown("<br>", unsafe_allow_html=True)

    # Display anomalies for the selected site
    st.subheader(f'Detected Anomalies in {selected_site}')
    anomalies, mean_value, threshold = detect_anomalies(filtered_data, col_prefix)
    st.write(anomalies)
    st.markdown("<br>", unsafe_allow_html=True)

    # Add button to generate predictions
    if st.button('Generate Predictions'):
        predictions_df = generate_predictions(anomalies, selected_site, col_prefix, mean_value, threshold)
        st.subheader(f'Predictions for {selected_site}')
        
        # Use st.markdown to render HTML
        for i, row in predictions_df.iterrows():
            st.markdown(f"""
            <table>
                <tr>
                    <th>Date</th>
                    <td>{row['Date']}</td>
                </tr>
                <tr>
                    <th>Consumption</th>
                    <td>{row['Consumption']}</td>
                </tr>
                <tr>
                    <th>Unusually High/Low</th>
                    <td>{row['Unusually High/Low']}</td>
                </tr>
                <tr>
                    <th>Causes</th>
                    <td>{row['Causes']}</td>
                </tr>
                <tr>
                    <th>Predictions</th>
                    <td>{row['Predictions']}</td>
                </tr>
            </table>
            <br>
            """, unsafe_allow_html=True)

elif nav_option == 'Water Quality':
    st.sidebar.header('Water Quality Options')
    quality_option = st.sidebar.selectbox('Select Water Quality Parameter', ['Scaling (LSI)', 'pH', 'TDS'])

    if quality_option == 'Scaling (LSI)':
        st.subheader('Scaling (LSI)')
        st.markdown("")
        # Plot LSI values with categories
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=water_quality_data['Date'], y=water_quality_data['LSI'], hue=water_quality_data['LSI Category'], dodge=False, palette={
            'Highly Corrosive': 'red',
            'Moderately Corrosive': 'orange',
            'Slightly Corrosive to Balanced': 'yellow',
            'Slightly Scaling': 'lightgreen',
            'Moderately Scaling': 'green',
            'Highly Scaling': 'darkgreen'
        }, ax=ax)
        ax.set_title('LSI Values Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('LSI Value')
        ax.legend(title='LSI Category')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("")
        st.markdown("""
        ### Interpretation of LSI Values
        - **LSI < -2: Highly Corrosive** - The water has a high tendency to dissolve calcium carbonate, leading to corrosion of pipes and equipment.
        - **LSI between -2 and -0.5: Moderately Corrosive** - The water is somewhat corrosive and may cause some dissolution of calcium carbonate.
        - **LSI between -0.5 and 0: Slightly Corrosive to Balanced** - The water is either slightly corrosive or balanced, with a tendency to neither dissolve nor precipitate calcium carbonate.
        - **LSI between 0 and 0.5: Slightly Scaling** - The water is slightly supersaturated with calcium carbonate, with a tendency to form a protective scale layer.
        - **LSI between 0.5 and 2: Moderately Scaling** - The water has a tendency to form scale, and scaling is likely to occur.
        - **LSI > 2: Highly Scaling** - The water is highly supersaturated with calcium carbonate, indicating a strong tendency to form scale rapidly.
        """)

    if quality_option == 'pH':
        st.subheader('pH Levels')

    # Plot pH values with categories
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=water_quality_data['Date'], y=water_quality_data['pH'], hue=water_quality_data['pH Category'], dodge=False, palette={
            'Acidic': 'red',
            'Normal': 'green',
            'Alkaline': 'blue'
        }, ax=ax)
        ax.set_title('pH Values Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('pH Value')
        ax.legend(title='pH Category')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("""
        ### Impact of pH Outliers on a Water Management Circuit
        """)

        # Add button to generate pH impact predictions
        if st.button('Generate pH Impact Predictions'):
            predictions = []
            for _, row in water_quality_data.iterrows():
                ph_category = categorize_ph(row['pH'])
                if ph_category == 'Acidic':
                    impact = """
                    <ul>
                    <li><strong>Corrosion</strong>: Accelerated corrosion of metal pipes, tanks, heat exchangers, and other equipment. Potential damage to protective coatings and linings inside pipes and tanks. Increased risk of leaks and equipment failures due to thinning and pitting of metal surfaces.</li>
                    <li><strong>Operational Efficiency</strong>: Corrosion products can clog and foul equipment, reducing heat transfer efficiency and increasing wear and tear on moving parts.</li>
                    <li><strong>Chemical Reactions</strong>: Reduced effectiveness of chemical scale inhibitors and need for higher dosages of corrosion inhibitors and other treatment chemicals.</li>
                    </ul>
                    """
                    predictions.append([row['Date'], row['pH'], ph_category, impact])
                elif ph_category == 'Alkaline':
                    impact = """
                     <ul>
                    <li><strong>Precipitation of Minerals</strong>: Increased tendency for calcium carbonate and magnesium hydroxide to precipitate, causing scaling. Potential for silica scaling if silica concentrations are high.</li>
                    <li><strong>Microbial Growth</strong>: Enhanced growth of biofilms and microbial colonies, which can clog pipes and reduce efficiency. Some bacteria can cause microbial-induced corrosion (MIC).</li>
                    <li><strong>Chemical Reactions</strong>: Reduced effectiveness of chlorine and other disinfectants. Altered efficiency of coagulation and flocculation processes in water treatment.</li>
                    </ul>
                    """
               
                    predictions.append([row['Date'], row['pH'], ph_category, impact])
            
            predictions_df = pd.DataFrame(predictions, columns=["Date", "pH", "pH Category", "Impact"])
            
            for i, row in predictions_df.iterrows():
                st.markdown(f"""
                <table>
                    <tr>
                        <th>Date</th>
                        <td>{row['Date']}</td>
                    </tr>
                    <tr>
                        <th>pH</th>
                        <td>{row['pH']}</td>
                    </tr>
                    <tr>
                        <th>pH Category</th>
                        <td>{row['pH Category']}</td>
                    </tr>
                    <tr>
                        <th>Impact</th>
                        <td>{row['Impact'].replace('\n', '<br>')}</td>
                    </tr>
                </table>
                <br>
                """, unsafe_allow_html=True)



    elif quality_option == 'TDS':
        st.subheader('Total Dissolved Solids (TDS)')
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set the background color to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot the data
        ax.plot(water_quality_data['Date'], water_quality_data['TDS'], label='TDS', color='blue')
        ax.axhline(y=1000, color='red', linestyle='--', label='Threshold (1000 mg/L)')

        # Set the title and labels with white color
        ax.set_title('TDS Values Over Time', color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('TDS (mg/L)', color='white')

        # Set the tick parameters with white color
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Set the legend with white color
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color('white')

        plt.xticks(rotation=45)
        st.pyplot(fig)
        if st.button('Generate TDS Impact '):
          st.markdown("""
        ### High TDS Impact Predictions
        - **Heat Exchangers and Boilers**
            - **Impact**: Scaling: High TDS leads to the precipitation of dissolved minerals like calcium and magnesium, forming scale deposits on heat exchange surfaces.
            - **Consequences**:
                - Reduced Heat Transfer Efficiency: Scale acts as an insulator, reducing the efficiency of heat exchange processes.
                - Increased Energy Consumption: More energy is required to achieve the same level of heating, leading to higher operational costs.
                - Overheating and Damage: Accumulated scale can cause hotspots, leading to overheating and potential damage to the equipment.
        - **Pipes and Valves**
            - **Impact**: Scaling and Fouling: High TDS water promotes the buildup of scale inside pipes and on valve surfaces.
            - **Consequences**:
                - Flow Restrictions: Scale buildup narrows the internal diameter of pipes, reducing flow rates and increasing pressure drops.
                - Increased Wear and Tear: Deposits can cause abrasion and increase mechanical wear on moving parts in valves.
                - Blockages: Severe scaling can lead to partial or complete blockages, requiring frequent cleaning and maintenance.
        - **Cooling Towers**
            - **Impact**: Scaling and Biological Growth: High TDS levels can enhance scaling on cooling tower fill material and promote biological growth.
            - **Consequences**:
                - Reduced Cooling Efficiency: Scale deposits reduce the effectiveness of evaporative cooling, leading to higher operating temperatures.
                - Increased Maintenance: More frequent cleaning and chemical treatments are required to control scale and biological fouling.
                - Microbial Corrosion: Biofilms can lead to microbial-induced corrosion (MIC), damaging cooling tower components.
        - **Recirculating Water Systems**
            - **Impact**: Corrosion: Certain dissolved solids, such as chlorides and sulfates, can be highly corrosive, especially at high concentrations.
            - **Consequences**:
                - Metal Degradation: Accelerated corrosion of metal pipes, tanks, and other components in the recirculating system.
                - Leaks and Failures: Increased risk of leaks due to thinning and pitting of metal surfaces, potentially leading to system failures and downtime.
                - Contamination: Corrosion byproducts can contaminate the recirculating water, affecting downstream processes.
        - **Water Treatment Equipment**
            - **Impact**: Increased Load: High TDS levels increase the load on water treatment equipment like softeners, reverse osmosis (RO) units, and demineralizers.
            - **Consequences**:
                - Frequent Regeneration: Water softeners and ion exchangers require more frequent regeneration cycles to handle high TDS levels.
                - Reduced Efficiency: RO membranes and other filtration units can become fouled more quickly, reducing their efficiency and lifespan.
                - Higher Operational Costs: Increased use of chemicals and energy for water treatment processes.
        """)