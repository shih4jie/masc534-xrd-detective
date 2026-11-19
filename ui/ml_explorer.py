"""ML-focused presentation; original physics remains available independently."""
import json
import numpy as np
import pandas as pd
import streamlit as st
from ml.predictor import predict_lpa_nn, reproduce_original_measurement
from ml.compatibility import model_requirements
from xrd.io import read_pattern


def render_ml_explorer(root, line_plot):
    st.subheader('LPA-NN Explorer')
    st.caption('Original trained inference, model compatibility and validation evidence. No prediction uncertainty is estimated.')
    model = st.selectbox('LPA-NN model', ['Spinel — original trained checkpoint',
        'Size — monoclinic ZrO2; weights unavailable', 'Size_strain — monoclinic ZrO2; weights unavailable'], key='ml_model')
    source = st.radio('ML input source', ['Upload XRD', 'Original Spinel reproduction'], key='ml_source')
    # Display only results of the explicit current action. Any edit clears them.
    if model != 'Spinel — original trained checkpoint':
        st.info('This module contains training notebooks, not a released checkpoint. No randomly initialized prediction is available. Use the physics analyzer while training remains pending.')
    elif source == 'Upload XRD':
        st.warning('No new experimental upload domain has passed held-out accuracy validation. Compatibility checks are available; new-upload predictions remain withheld.')
        uploaded = st.file_uploader('ML pattern (two columns: 2theta degrees, intensity)',type=['csv','txt','xy','dat'],key='ml_upload')
        wl = st.number_input('ML measurement wavelength (nm)',min_value=.000001,value=.154,format='%.6f',key='ml_wavelength')
        verified = st.checkbox('I verified this wavelength and degree angular units',key='ml_verified')
        geometry = st.selectbox('ML measurement geometry',['Unknown / not declared','Powder θ–2θ scan','Capillary powder measurement','Other'],key='ml_geometry')
        phases = st.text_input('ML independently identified phases (comma separated)',key='explorer_phases')
        instrument = st.text_input('ML instrument / beamline',key='explorer_instrument')
        if uploaded is not None:
            try:
                pattern = read_pattern(uploaded.getvalue(),source=uploaded.name)
                result = predict_lpa_nn(pattern, {'wavelength_nm':wl if verified else None,'geometry':geometry,
                    'phases':[v.strip() for v in phases.split(',') if v.strip()], 'instrument':instrument})
                st.write(result['input_compatibility_status'])
                st.dataframe(pd.DataFrame(result['compatibility_checks']),hide_index=True,width='stretch')
                line_plot(pattern.two_theta_deg,{'Uploaded intensity':pattern.intensity},uploaded.name)
            except (ValueError,UnicodeError,TypeError,OverflowError):
                st.error('Cannot read this pattern. Export two numeric columns with increasing 2theta in degrees and finite intensities. Use Analyze My XRD for explicit extra-column selection.')
        st.button('Run ML prediction',disabled=True,key='ml_run_upload')
        st.caption('Use Analyze My XRD for physics analysis under its own calibration and broadening assumptions.')
    else:
        st.warning('Research reproduction, not validated prediction for a new sample. These original experimental rows have no independent size labels.')
        index = int(st.slider('Original Spinel measurement index',0,125,0,key='ml_original_index'))
        folder=root/'third_party/LPA-NN/Spinel'
        available=all((folder/name).is_file() for name in ['CNN.hdf5','normL.class','data.npz'])
        if not available:
            st.info('Original assets are missing. Follow third_party/README.md to obtain the pinned upstream checkout; no substitute weights are used.')
        if st.button('Run original trained model',type='primary',disabled=not available,key='ml_reproduce'):
            with st.status('Loading verified checkpoint and running original inference…',expanded=True) as status:
                try:
                    result=reproduce_original_measurement(index)
                    status.update(label='Original inference complete — accuracy unvalidated',state='complete',expanded=False)
                    st.warning('Unvalidated research outputs. Do not interpret these values as independent size or microstrain measurements.')
                    st.metric('Original model size output',f"{result['predicted_size_nm']:.4f} nm")
                    st.metric('Original model microstrain output',f"{result['predicted_microstrain_percent']:.5f} %")
                    with np.load(folder/'data.npz',allow_pickle=False) as archive:
                        row=archive['exp_data'][index]
                    line_plot(result.pop('plotting_axis_deg'),{'Original experimental intensity':row},'LPA-NN Spinel/data.npz; plotting grid only')
                    st.json(result)
                except (ImportError,ValueError,OSError,RuntimeError,TypeError):
                    status.update(label='Inference unavailable',state='error',expanded=False)
                    st.error('Original checkpoint could not be verified or executed. Check pinned asset hashes and requirements-ml.txt; physics analysis remains available.')
    with st.expander('Model compatibility and training and validation results'):
        st.json(model_requirements())
        st.write('Size and Size_strain: original residual CNN, label columns 20 and 20/22, MinMax targets, Adam 1e-4, batch 32, example 5 epochs, 10% validation from training only. No training is claimed here.')
        st.write('Held-out Spinel sample count: 0. MAE, RMSE, relative errors and size-range accuracy are unavailable. The 126 original rows establish numerical execution only.')
        for name in ['spinel_reproduction.json','spinel_benchmark_status.json']:
            p=root/'validation/results'/name
            if p.exists():
                # Cloud-evicted evidence must not block independent physics UI.
                if getattr(p.stat(),'st_flags',0) & 0x40000000:
                    st.warning(f'Saved evidence {name} is cloud-only. Download it to inspect this record; physics analysis remains available.')
                else:
                    st.json(json.loads(p.read_text()))
    st.info('Synthetic exploration below is a single-peak physics demonstration. A compatible full-pattern generator and instrument configuration are absent, so these simulated peaks cannot be sent to the Spinel CNN.')
