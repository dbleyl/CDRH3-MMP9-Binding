#import tensorflow.compat.v1.keras.backend as K
#import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
#!pip install shap
def shap_plots(model, test_ab_array, df_test_ab_shuffled, result):
    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()
    from tensorflow import keras
    import numpy as np
    import pickle
    import streamlit as st
    if model == 'ESM2 650MB':
        lstm_model = keras.models.load_model("/content/CDRH3-MMP9-Binding/saved-representations-and-models/lstm_650MB_run1_SHAP_Mask_revised_mmp9_030824")
        with open('/content/x_training_650MB_array_run3_test_ab.pkl', 'rb') as f:
            X_training_array = pickle.load(f)
    elif model == 'ESM2 3B':
        lstm_model = keras.models.load_model("/content/CDRH3-MMP9-Binding/saved-representations-and-models/lstm_3B_run1_SHAP_revised_mmp9_030824")
        test_ab_array = representations_3B(df_shuffled_list,df_binding_AA_shuffled_list)
    elif model == 'ESM2 15B':
        lstm_model = keras.models.load_model("/content/CDRH3-MMP9-Binding/saved-representations-and-models/lstm_15B_SHAP_run1_revised_mmp9_030824")
        test_ab_array = representations_15B(df_shuffled_list,df_sequence_shuffled_list)
    else:
        lstm_model = keras.models.load_model("/content/CDRH3-MMP9-Binding/saved-representations-and-models/lstm_antiberty_run1_SHAP_Mask_revised_mmp9_030824")
        with open('/content/x_training_antiberty_array_run3_test_ab.pkl', 'rb') as f:
            X_training_array = pickle.load(f)


    #print("test shape", X_test_650MB_array.shape)
    #print("training array shape", X_training_array.shape )
    st.write("training array shape", X_training_array.shape )
    st.write("test shape", test_ab_array.shape)

    import shap

    explainer = shap.DeepExplainer(lstm_model, X_training_array)
    shap_values1 = explainer.shap_values(test_ab_array, check_additivity=False)
    import matplotlib
    shap.initjs()
    st.write("Model:", model)
    for i in range(len(df_test_ab_shuffled)):
        CDRH3_String= df_test_ab_shuffled.iloc[i,4]
        binding = result.iloc[i,1]
        st.write(str(i+1) + " CDRH3: ", CDRH3_String + ", Predicted Binding: ", binding)
        cdrh3_aa_list = []
        m=1
        for aa in CDRH3_String:
            cdrh3_aa_list.append(aa + str(m))
            m=m+1
        array_shap_values1 = np.array(shap_values1[i])
        array_shap_values1_sum=np.sum(array_shap_values1[0:len(CDRH3_String)], axis=1)
        array_shap_values1_sum_reduced = array_shap_values1_sum[:,0]
        #st.pyplot((shap.force_plot(explainer.expected_value[0], array_shap_values1_sum_reduced,cdrh3_aa_list,matplotlib=matplotlib)))
        fig = (shap.force_plot(explainer.expected_value[0], array_shap_values1_sum_reduced,cdrh3_aa_list,matplotlib=matplotlib))
        st.pyplot(fig)

