import utils.preprocess as pp
import utils.settings as st


def main():
    data = pp.prepare_dataset(
        st.DATASET_PATH,
        st.TEST_SAMPLE_PATH,
        st.METRIC_NAME
    )
    data = pp.preprocess_variables(data,
                                   st.IMPUTER_PATH)
    pp.split_dataset(data,
                     st.TEST_SIZE,
                     st.RANDOM_STATE,
                     st.METRIC_NAME,
                     st.TRAIN_DATASET_PATH,
                     st.TEST_DATASET_PATH)

if __name__ == "__main__":
    main()