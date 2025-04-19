from dataset import download_dataset, format_for_logistic_regression, load_data
from networks import classification_network
from training import train_network


def main():
    download_dataset()
    format_for_logistic_regression()
    model = classification_network()
    train_network(model)
    # generate synthetic data
    # train networks
    # test networks
    # test using sdr data
    # compare networks


if __name__ == "__main__":
    main()
