import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("min_support", type=float,
                    help='min support')
parser.add_argument("load_file_path", type=str,
                    help='load file path')
parser.add_argument("save_file_path", type=str,
                    help='save file path')
args = parser.parse_args()


header_table = transactions = fp_tree_root = None


class Node:
    def __init__(self, parent_node=None, value=None):
        self.parent = parent_node
        self.value = value
        self.node_link = None
        self.counts = 1
        self.children = dict()  # value to node dict

    def add_one_count(self):
        self.counts += 1


def construct_header_table():
    global header_table, transactions

    # construct header table and transactions array with items which supports are greater than min support
    with open(args.load_file_path, "r") as fp:
        lines = fp.readlines()
        transactions = list()
        header_table = dict()  # (item: (frequency, fp_tree_pointer))
        for line in lines:
            transaction = np.array(line.strip().split(','))
            transactions.append(transaction)
            for item in transaction:
                header_table[item] = [1, None] if item not in header_table else [header_table[item][0] + 1, None]
        transactions = np.array(transactions)
        header_rows = sorted(header_table.items(), key=lambda x: x[1][0], reverse=True)
        saved_idx = 0
        for i in range(len(header_rows)):
            item_freq = header_rows[i][1][0] / len(transactions)
            header_rows[i][1][0] = item_freq
            if item_freq < args.min_support:
                break
            saved_idx += 1
        header_table = dict(np.array(header_rows)[:saved_idx])

    # reconstruct transactions by header table
    for i in range(len(transactions)):
        saved_items = [item for item in transactions[i] if item in header_table.keys()]
        transactions[i] = np.array(sorted(saved_items, key=lambda x: header_table[x][0], reverse=True))


def DFS(cur_node):
    print(f'at item: {cur_node.value}\tcounts: {cur_node.counts}')
    if not cur_node.children:
        return
    for item in cur_node.children:
        next_node = cur_node.children[item]
        DFS(next_node)


def construct_fp_tree():
    global transactions, fp_tree_root
    item2prevNode_dict = dict([(item, None) for item in header_table.keys()])
    fp_tree_root = Node()
    for transaction in transactions:
        cur_node = fp_tree_root
        print(f'transaction: {transaction}')
        for item in transaction:
            if item in cur_node.children:  # there is a node corresponding to value in children, so add the counts of it
                print(f'\tcur-node: {cur_node.value}\t[find corresponding node, add one count]: {item}')
                corr_node = cur_node.children[item]
                corr_node.add_one_count()
                cur_node = corr_node
            else:  # add new node
                new_node = Node(parent_node=cur_node, value=item)
                if item2prevNode_dict[item] is None:  # the first value to appear, so set the header table
                    header_table[item][1] = new_node
                    print(f'\tcur-node: {cur_node.value}\t[add-node | init-header]: {item}')
                else:
                    item2prevNode_dict[item].node_link = new_node
                    print(f'\tcur-node: {cur_node.value}\t[add-node | update-dict]: {item}')
                item2prevNode_dict[item] = new_node
                cur_node.children[item] = new_node
                cur_node = new_node


def fp_growth():
    construct_header_table()
    construct_fp_tree()
    DFS(fp_tree_root)


def main():
    fp_growth()


if __name__ == '__main__':
    main()
