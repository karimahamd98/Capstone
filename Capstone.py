#Importing required packages
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
from streamlit_option_menu import option_menu
import plotly.express as px
from statistics import  mean
import hydralit_components as hc
import time
from sklearn.ensemble import RandomForestRegressor





#Importing the data
df1 = pd.read_csv('https://raw.githubusercontent.com/karimahamd98/Capstone/main/Msba_Data.csv')
df= pd.read_csv('https://raw.githubusercontent.com/karimahamd98/Capstone/main/MSBA%20Data.csv')


#Page layout

st.set_page_config(
    page_title="MSBA Students Performance Analytics",
    page_icon="🎓",
    layout="wide",
)


# Option menu for Navigation

selected = option_menu(None, ["Home","Overview","Performance Prediction"],
    icons=['house',"eye","skip-forward-fill"],
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "5!important", "background-color": "#6d021b"},
        "icon": {"color": "white", "font-size": "25px"},
        "nav-link": {"font-size": "22px", "text-align": "center","color":"white", "margin":"0px", "--hover-color": "#a14054"},
        "nav-link-selected": {"background-color": "#d50032"},
    })


#If statment to turn the buttons into functional elements
if selected == 'Home':
    st.markdown(f'<h1 style="color:#6d021b;font-size:38px;">{"MSBA Student Perfromance Analysis & Prediction"}</h1>', unsafe_allow_html=True)

    #Splitting the page into columns to position elements as desired
    col1,col2,col3 = st.columns([5,1,4])

    col1.text_area('Project Summary','''
This Capstone Project constitutes of 2 main parts:

◈ Streamlit App:
    - Exploratory Data Analysis page with dynamic & interactive plots
    - Performance prediction page, a model that will predict incoming students GPA based on
      several factors

◈ PowerBI:
    - A dashboard that provides a broader view into the MSBA data
''', height = 310)


    with col3:
        #The 'st.text' is to create blank space before the image
        st.text("")
        st.text("")
        st.text("")
        st.text("")

        image = 'https://raw.githubusercontent.com/karimahamd98/Capstone/main/logo_aub.png'
        col3.image(image, width=555)



#If statment to turn the buttons into functional elements
if selected == "Overview":

        # Values for the info cards

        value1 = mean(df1['GPA'])
        value2 = "{:.2f}".format(value1)
        value3 = mean(df1['Age_when_Joined_Program'])
        value4 = "{:.0f}".format(value3)
        value5 = df.Status.value_counts()
        value6 = "{:.0f}".format(value5[0])
        value7 = "{:.0f}".format(value5[1])



        # Splitting the page

        info = st.columns(4)
        col1,col2= st.columns([8,2])


        # Setting a theme for each cards

        theme1 = {'bgcolor': '#f9f9f9','title_color': '#6d021b','content_color': '#6d021b','icon_color': '#6d021b','icon': 'bi-file-earmark-binary'}
        theme2 = {'bgcolor': '#f9f9f9','title_color': '#6d021b','content_color': '#6d021b','icon_color': '#6d021b','icon': 'bi-person-bounding-box'}
        theme3 = {'bgcolor': '#f9f9f9','title_color': '#6d021b','content_color': '#6d021b','icon_color': '#6d021b','icon': 'bi-people'}
        theme4 = {'bgcolor': '#f9f9f9','title_color': '#6d021b','content_color': '#6d021b','icon_color': '#6d021b','icon': 'bi-people-fill'}





        #Displaying the cards
        with info[0]:
            hc.info_card(title= 'Average Student GPA', content = value2, theme_override = theme1)

        with info[1]:
            hc.info_card(title = 'Average Student Age',content= value4 , theme_override = theme2)

        with info[2]:
            hc.info_card(title = 'MSBA Alumni',content= value6 , theme_override = theme3)


        with info[3]:
            hc.info_card(title = 'Active Students',content= value7 , theme_override = theme4)

            with col1:
                st.markdown(f'<h1 style="color:#6d021b;font-size:24px;">{"Glance at Dataset"}</h1>', unsafe_allow_html=True)
                #If statment to show or hide data with variable input to set how many columns to show
                st.markdown("")
                show = st.radio("View Data?",["Hide","Show"])
                if show == "Hide":
                    st.write("")
                if show == 'Show':

                    col_count=st.selectbox('How Many Columns to Show:',[1,5,10,20])
                    table = df.head(col_count)
                    st.table(table)


        st.markdown(f'<h1 style="color:#6d021b;font-size:22px;">{"Average GPA Visualized"}</h1>', unsafe_allow_html=True)
        #Select box elements that will be used as inputs for the visualization
        variable= st.selectbox('Average GPA by:',['School','Last_Degree','Background','Cohort','Decision'])

        Cohort_input= st.selectbox('Cohort Selection:',['All Cohorts','Cohort 1','Cohort 2','Cohort 3','Cohort 4', 'Cohort 5'])

        c = df1.loc[df1['Cohort'] == Cohort_input]
        #If statments to show visuals in accordance to the input
        #Using plotly express to visualize
        if Cohort_input == 'Cohort 1':
            fig_c1 = px.histogram(c, x=c.GPA , y=variable,
                    histfunc='avg',color_discrete_sequence=['#6d021b']).update_yaxes(categoryorder="total ascending")

            st.plotly_chart(fig_c1,use_container_width = True)
        elif Cohort_input == 'Cohort 2':

            fig_c2 = px.histogram(c, x=c.GPA , y=variable,
                    histfunc='avg',color_discrete_sequence=['#6d021b']).update_yaxes(categoryorder="total ascending")

            st.plotly_chart(fig_c2,use_container_width = True)

        elif Cohort_input == 'Cohort 3':

            fig_c3 = px.histogram(c, x=c.GPA , y=variable,
                    histfunc='avg',color_discrete_sequence=['#6d021b']).update_yaxes(categoryorder="total ascending")

            st.plotly_chart(fig_c3,use_container_width = True)


        elif Cohort_input == 'Cohort 4':

            fig_c4 = px.histogram(c, x=c.GPA , y=variable,
                    histfunc='avg',color_discrete_sequence=['#6d021b']).update_yaxes(categoryorder="total ascending")

            st.plotly_chart(fig_c4,use_container_width = True)


        elif Cohort_input == 'Cohort 5':

            fig_c5_best_cohort_ever = px.histogram(c, x=c.GPA , y=variable,
                    histfunc='avg',color_discrete_sequence=['#6d021b']).update_yaxes(categoryorder="total ascending")

            st.plotly_chart(fig_c5_best_cohort_ever,use_container_width = True)


        elif Cohort_input == 'All Cohorts':
            fig = px.histogram(df1, x=df1.GPA , y=variable,
                histfunc='avg',color_discrete_sequence=['#6d021b']).update_yaxes(categoryorder="total ascending")

            st.plotly_chart(fig,use_container_width = True)
        else:
            fig1 = px.histogram(df1, x=df1.GPA , y=variable,
                histfunc='avg',color_discrete_sequence=['#6d021b']).update_yaxes(categoryorder="total ascending")

            st.plotly_chart(fig1,use_container_width = True)

        ozr=st.checkbox('Other Visuals')
        # If statment to show the same plot in other visualizations (Pie chart, Tree map)
        if ozr:
            more=st.radio("",['Pie Chart', 'Tree Map'])

            if more == 'Pie Chart':

                if Cohort_input != 'All Cohorts':
                    dset = df1.groupby([variable,'Cohort'],as_index=False)['GPA'].mean()
                    c2 = dset.loc[dset['Cohort'] == Cohort_input]
                    figz = px.pie(c2, values=c2.GPA, names=variable, color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(figz,use_container_width = True)
                else:
                    dset = df1.groupby([variable,'Cohort'],as_index=False)['GPA'].mean()
                    figzz= px.pie(dset, values=dset.GPA, names=variable, color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(figzz,use_container_width = True)

            if more == 'Tree Map':

                if Cohort_input != 'All Cohorts' and variable != 'Cohort':
                    dset = df1.groupby([variable,'Cohort'],as_index=False)['GPA'].mean()
                    c2 = dset.loc[dset['Cohort'] == Cohort_input]
                    fig_tm= px.treemap(c2,path=[variable],values=c2.GPA,color=c2.GPA,color_continuous_scale='OrRd')
                    st.plotly_chart(fig_tm,use_container_width = True)
                else:
                    dset = df1.groupby([variable,'Cohort'],as_index=False)['GPA'].mean()
                    fig_tma= px.treemap(dset,path=[variable],values=dset.GPA,color=dset.GPA,color_continuous_scale='OrRd')
                    st.plotly_chart(fig_tma,use_container_width = True)


#The predictive part
if selected == "Performance Prediction":
    col1,col2,col3 = st.columns([5,1,2])


    with col1:

        st.write("")
        st.write("")
        st.markdown(f'<h1 style="color:#6d021b;font-size:22px;">{"Please Insert the Following Information:"}</h1>', unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        #Select box elements for the use to specify the variables that the model will predict based on
        gender= st.selectbox('Gender',['M','F'])
        category= st.selectbox('Full time or Part time',['FT','PT'])
        background= st.selectbox('Background',['Business/Economics/Accounting/Finance','Engineering','Math/science',
        'MIS/Computer Science/IT','Other (Food science, Nursing, Graphic Design..)'])
        uni= st.selectbox('University',["Al-Qasimia University","American University of Beirut (AUB)","American University of Sharjah","Arab Open University (AOU)","Arts, Sciences and Technology University in Lebanon (AUL)","Beirut Arab University (BAU)","Damascus University"
        ,"ESA Business School","Florida Int'l University, Miami","Grenoble Graduate School of Business","Haigazian University","Inst.Sup.des Et.Tech.en Comm.","Lebanese American University (LAU)","Lebanese International University (LIU)","Lebanese University (LU)","Notre Dame University Louaize (NDU)"
        ,"Rafic Hariri University (RHU)","Uni. of Wisconsin, Eau Claire \ Ec.Nat.de Com.et de Gest. ENCG","Universite Saint-Esprit de Kaslik (USEK)","Universite Saint-Joseph (USJ)","University of Balamand (UOB)","University of Maryland"])

        age=st.number_input("Age")

        #Dictionaries to assing each category to a label (Label Encoding)
        uni_dict= {"Al-Qasimia University":1,"American University of Beirut (AUB)":2,"American University of Sharjah":3,"Arab Open University (AOU)":4,"Arts, Sciences and Technology University in Lebanon (AUL)":5,"Beirut Arab University (BAU)":6,"Damascus University":7
        ,"ESA Business School":8,"Florida Int'l University, Miami":9,"Grenoble Graduate School of Business":10,"Haigazian University":11,"Inst.Sup.des Et.Tech.en Comm.":12,"Lebanese American University (LAU)":13,"Lebanese International University (LIU)":14,"Lebanese University (LU)":15,"Notre Dame University Louaize (NDU)":16
        ,"Rafic Hariri University (RHU)":17,"Uni. of Wisconsin, Eau Claire \ Ec.Nat.de Com.et de Gest. ENCG":18,"Universite Saint-Esprit de Kaslik (USEK)":19,"Universite Saint-Joseph (USJ)":20,"University of Balamand (UOB)":21,"University of Maryland":22}


        background_dict = {
         'Business/Economics/Accounting/Finance': 1,
         'Engineering': 2,
         'Math/science': 3,
         'MIS/Computer Science/IT': 4,
         'Other (Food science, Nursing, Graphic Design..)': 5}

        gender_dict = {"F":0,"M":1}
        catg_dict = {"PT":0,"FT":1}


        #Mapping the data to the dictionaries that have the encoding
        df1['Gender'] = df1['Gender'].map(gender_dict)
        df1['Category'] = df1['Category'].map(catg_dict)
        df1['Background'] = df1['Background'].map(background_dict)
        df1['School'] = df1['School'].map(uni_dict)


        #Splitting our Features & Target
        X= df1[['Gender', 'Category', 'Background','Age_when_Joined_Program','School' ]]
        y = df1['GPA']



        X = pd.DataFrame(X,columns=X.columns)
        y = df1['GPA']
        #Splitting the data into test & train sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        #RandomForestRegressor model
        rf = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 5, random_state = 18).fit(X_train, y_train)
        prediction = rf.predict(X_test)





        #Funtion to get the value (encoded label) from the dictionaries
        def get_value(val,my_dict):
            for key,value in my_dict.items():
                if val == key:
                    return value
        #Applying the function
        gndr = get_value(gender,gender_dict)
        categr = get_value(category,catg_dict)
        bckgrnd = get_value(background,background_dict)
        schl = get_value(uni,uni_dict)
        single_sample = [gndr,categr,bckgrnd,schl,age]
        #Predicting based on the newly inputed features
        if st.button("Predict"):
            with hc.HyLoader('Loading',hc.Loaders.standard_loaders,index=[6]):
                time.sleep(3)

            sample = np.array(single_sample).reshape(1,-1)
            prediction =rf.predict(sample)
            st.info("Predicted Potential GPA (0.2 RMSE)")
            st.header("{:.2f}".format(prediction[0]))


        with col3:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")

            image = 'https://cdn-icons-png.flaticon.com/512/2247/2247890.png'
            col3.image(image, use_column_width= True)
