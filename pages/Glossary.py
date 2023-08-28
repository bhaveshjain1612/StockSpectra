import streamlit as st
from utils import * 

st.set_page_config(page_title="StockSpectra - Glossary", layout = "wide")

def reorder_keys(data_dict, element):
    if element not in data_dict:
        return sorted(data_dict.keys())

    # Create a list with the element followed by other keys
    keys_list = [element] + [key for key in data_dict if key != element]
    return keys_list


#glossary function
def glossary(indicators,key=''):
    
    st.title("StockSpectra")
    st.header("Glossary")

    selected_indicator = st.selectbox("Select an Indicator", reorder_keys(indicators,key))
    st.divider()
    # Display the details of the selected indicator
    st.write("###", indicators[selected_indicator]["full_name"])
    st.write("**What is it?**")
    st.write(indicators[selected_indicator]["description"])
    st.write("**How does it work?**")
    st.write(indicators[selected_indicator]["how_it_works"])
    st.write("**Why is it useful?**")
    st.write(indicators[selected_indicator]["usefulness"])
    try:
        st.write("**How is it calculated?**")
        st.latex(indicators[selected_indicator]["calculation"])
    except:
        st.write("No calculations")
    x = 'https://www.investopedia.com/search?q='+indicators[selected_indicator]["full_name"].replace(" ","+")
    st.write(f"For more details, check out [this Link]( {x} )")


#main function
def main():
# Create a select box in the Streamlit app    
    indicatordetails = get_indicatordetails()

    if st.experimental_get_query_params() != {}:
        item = st.experimental_get_query_params()['item'][0].replace("_"," ")
        glossary(indicatordetails,item)
    else:
        glossary(indicatordetails,)
    st.experimental_set_query_params()
           
# Run the app
if __name__ == "__main__":
    main()
