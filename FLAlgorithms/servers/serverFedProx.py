from FLAlgorithms.users.userFedProx import UserFedProx
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data

class FedProx(Server):
    def __init__(self, args, model, seed, user_datasets=None):
        super().__init__(args, model, seed)

        if user_datasets is not None:
            print("Using externally provided user datasets.")
            self.users = []
            for i, data in enumerate(user_datasets):
                user = UserFedProx(args, i, model, data, data, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        else:
            data = read_data(args.dataset)
            total_users = len(data[0])
            print("Users in total: {}".format(total_users))
            for i in range(total_users):
                id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
                user = UserFedProx(args, id, model, train_data, test_data, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples

        print("Number of users / total users:", args.num_users, " / ", len(self.users))
        print("Finished creating FedProx server.")

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            # print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            self.send_parameters()
            self.evaluate()
            for user in self.selected_users:
                user.train(glob_iter)
            self.aggregate_parameters()
        self.save_results(args)
        self.save_model()
