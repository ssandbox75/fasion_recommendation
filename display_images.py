import streamlit as st
import pickle


filenames = pickle.load(open('file_names.pkl','rb'))

def displaying_images_in_pagination(displaying_images):
    # Number of entries per screen
    N = 5

    # Read the table and initialize page number to zero to view the first N entries in dataframe
    page_number, start_index, end_index = 0, 0, N
    # data = pd.read_csv("auto-mpg.csv")
    last_page = len(displaying_images) // N

    previous_page, _ ,next_page = st.columns([2, 4, 2])


    st.divider()
    column_list = list(st.columns(len(displaying_images)))

    if displaying_images:
        # If image not found
        for num in range(start_index, end_index):
            try:
                with column_list[num]:
                    st.image(
                        filenames[displaying_images[num]],
                        caption=f"{filenames[num].split('/')[1]}",
                        use_column_width=None
                    )
                    st.write(filenames[num].split('/')[1])
                    
            except IndexError as e:
                print('Only this products are available.')
                
    else:
        st.header("Product not found.")

    st.divider()
    if previous_page.button("Previous"):
        if page_number - 1 < 0:
            page_number = last_page
            start_index, end_index = 0, N
        else:
            page_number -= 1
            start_index -= N
            end_index -= N           
    if next_page.button("Next"):
        if page_number + 1 > last_page:
            page_number = 0
            start_index, end_index = 0, N
        else:
            page_number += 1
            start_index += N
            end_index += N