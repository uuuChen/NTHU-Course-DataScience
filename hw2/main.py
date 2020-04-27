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


class Node:
    def __init__(self, parent_node=None, value=None):
        self.parent = parent_node
        self.value = value
        self.link_node = None
        self.count = 1
        self.children = dict()  # value to node dict

    def add_count(self, count):
        self.count += count


def log_title(title_str, splin_len=40):
    print('\n' + '-' * splin_len + ' ' + title_str + ' ' + '-' * splin_len)


def load_file():
    log_title(f'load-file ({args.load_file_path})')
    with open(args.load_file_path, "r") as fp:
        lines = fp.readlines()
        trans2count = dict()  # (transaction: count)
        for line in lines:
            transaction = tuple(line.strip().split(','))
            trans2count.setdefault(transaction, 0)
            trans2count[transaction] += 1
        args.num_of_transactions = len(trans2count.keys())
        return trans2count


def construct_header_table(trans2count):
    log_title(f'construct-header-table')
    # construct header table and transactions array with items which supports are greater than min support
    header_table = dict()  # (item: (frequency, fp_tree_pointer))
    for transaction, count in trans2count.items():
        for item in transaction:
            header_table[item] = [count, None] if item not in header_table else [header_table[item][0] + count, None]
    header_rows = sorted(header_table.items(), key=lambda x: x[1][0], reverse=True)
    print(f'header rows: {header_rows}')
    saved_idx = 0
    for i in range(len(header_rows)):
        item_freq = header_rows[i][1][0] / args.num_of_transactions
        header_rows[i][1][0] = item_freq
        if item_freq < args.min_support:
            break
        saved_idx += 1
    if saved_idx == 0:  # all items are less than min support, return (None, None)
        print('all less than min support')
        return None, None
    header_table = dict(np.array(header_rows)[:saved_idx])
    print(f'header table: {header_table}')

    # reconstruct transactions by header table
    saved_trans2count = dict()
    for transaction, count in trans2count.items():
        saved_items = [item for item in transaction if item in header_table.keys()]
        saved_transaction = tuple(sorted(saved_items, key=lambda x: header_table[x][0], reverse=True))
        saved_trans2count[saved_transaction] = count

    return header_table, saved_trans2count


def construct_fp_tree(trans2count):
    header_table, saved_trans2count = construct_header_table(trans2count)
    log_title(f'construct-fp-tree')
    print('trans2count: ' + str(trans2count))
    print('saved_trans2count: ' + str(saved_trans2count))
    if not header_table:
        return None, None
    item2prevNode_dict = dict([(item, None) for item in header_table.keys()])
    fp_tree_root = Node()
    for transaction, count in saved_trans2count.items():
        cur_node = fp_tree_root
        print(f'transaction: {transaction}')
        for item in transaction:
            if item in cur_node.children:  # there is a node corresponding to value in children, so add the count of it
                corr_node = cur_node.children[item]
                corr_node.add_count(count)
                print(f'\tparent-node: {cur_node.value}\t[find corresponding node | count: {corr_node.count} '
                      f'| correspond item: {item}]')
                cur_node = corr_node
            else:  # add new node
                new_node = Node(parent_node=cur_node, value=item)
                if item2prevNode_dict[item] is None:  # the first value to appear, so set the header table
                    header_table[item][1] = new_node
                    print(f'\tparent-node: {cur_node.value}\t[add-node | init-header | count: {new_node.count} '
                          f'| new item: {item}]')
                else:
                    item2prevNode_dict[item].link_node = new_node
                    print(f'\tparent-node: {cur_node.value}\t[add-node | update-dict | count: {cur_node.count} '
                          f'| new item: {item}]')
                item2prevNode_dict[item] = new_node
                cur_node.children[item] = new_node
                cur_node = new_node
    return fp_tree_root, header_table


def get_prefix_trans2count(node):
    trans2count = dict()
    leaf_node = node
    while leaf_node is not None:  # until no next same value node
        transaction = list()
        cur_node = leaf_node
        while cur_node.value is not None:  # ascending until root
            transaction.append(cur_node.value)
            cur_node = cur_node.parent
        trans2count[tuple(reversed(transaction[1:]))] = leaf_node.count
        leaf_node = leaf_node.link_node
    return trans2count


def mine_fp_tree(header_table, prefix_freq_set, freq_item_list):
    log_title('mine-fp-tree')
    for item, (item_freq, node_pointer) in reversed(list(header_table.items())):
        print(f'\nin item {item}')
        new_freq_set = prefix_freq_set.copy()
        new_freq_set.add(item)
        print(f'new_freq_set: {new_freq_set}')
        freq_item_list.append((new_freq_set, item_freq))
        prefix_trans2count = get_prefix_trans2count(node_pointer)
        _, prefix_header_table = construct_fp_tree(prefix_trans2count)
        print(f'prefix_header_table: {prefix_header_table}')
        if prefix_header_table is not None:
            mine_fp_tree(prefix_header_table, new_freq_set, freq_item_list)


def depth_first_search(cur_node):
    print(f'at item: {cur_node.value}\tcount: {cur_node.count}')
    if not cur_node.children:
        return
    for item in cur_node.children:
        next_node = cur_node.children[item]
        depth_first_search(next_node)


def write_results2file(freq_pats):
    freq_pats = [[list(sorted(map(int, freq_pat[0]))), freq_pat[1]] for freq_pat in freq_pats]
    sorted_freq_pats = sorted(freq_pats, key=lambda item: (len(item[0]), item[0]))
    with open(args.save_file_path, "w") as fp:
        for freq_pat, freq in sorted_freq_pats:
            freq_pat_str = ",".join(map(str, freq_pat))
            freq = round(freq, 4)
            fp.write(freq_pat_str + f":{freq:.4f}\n")


def fp_growth():
    all_trans2count = load_file()
    fp_tree_root, fp_header_table = construct_fp_tree(all_trans2count)
    freq_item_list = list()
    mine_fp_tree(fp_header_table, prefix_freq_set=set([]), freq_item_list=freq_item_list)
    write_results2file(freq_item_list)


def main():
    fp_growth()


if __name__ == '__main__':
    main()
