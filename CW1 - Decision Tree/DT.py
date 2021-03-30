#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict 
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from copy import deepcopy
import os



def split_dataset(dataset,n,i,seed = 0):
    datasize = len(dataset)
    train_dataset = np.vstack((dataset[0:int(i*datasize/n)],dataset[int((i+1)*datasize/n):]))
    test_dataset = dataset[int(i*datasize/n):int((i+1)*datasize/n)]
    return train_dataset,test_dataset

class Node:
    def __init__(self,attribute=None,value=None,left=None,right=None,left_dataset=None,right_dataset=None,is_a_leaf=False,leaf_value=None,depth=None,parent=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.left_dataset = left_dataset
        self.right_dataset = right_dataset
        self.is_a_leaf = is_a_leaf
        self.leaf_value = leaf_value
        self.depth = depth
        self.position = None
        self.p_position = None
        self.parent = parent
        self.keep = False
        self.major_value = None

def get_major_value(dataset):
    count = defaultdict(lambda: 0)
    max_count=0
    max_label=None
    for data in dataset:
        count[data[-1]]+=1
    for label in count.keys():
        if count[label]>max_count:
            max_count = count[label]
            max_label = label
    return label
        
def decision_tree_learning(training_dataset,depth):
    if is_leaf(training_dataset):
        leaf_value = training_dataset[0,-1]
        return Node(is_a_leaf=True,leaf_value=leaf_value,depth=depth),depth
    else:
        major_value = get_major_value(training_dataset)   
        attribute,value,IG = find_split(training_dataset)
        left_dataset_filter = training_dataset[:,attribute] <= value
        right_dataset_filter = training_dataset[:,attribute] > value
        left_dataset = training_dataset[left_dataset_filter]
        right_dataset = training_dataset[right_dataset_filter]
        left_branch,left_depth = decision_tree_learning(left_dataset,depth+1)
        right_branch,right_depth = decision_tree_learning(right_dataset,depth+1)
        this_node = Node(attribute,value,left_branch,right_branch,left_dataset,right_dataset,depth=depth)
        this_node.major_value = major_value
        left_branch.parent = this_node
        right_branch.parent = this_node
        return this_node,max(left_depth,right_depth)
    
def is_leaf(training_dataset):
    for label in training_dataset[1:,-1]:
        if label != training_dataset[0,-1]:
            return False
    return True

def find_split(training_dataset):
    entropy = calc_entropy(training_dataset)
    dataset_size = training_dataset.shape[0]
    split_attribute = None
    split_value = None
    max_IG = 0
    for attribute in range(training_dataset.shape[1]-1):
        ordered_values = sorted(list(set(training_dataset[:,attribute])))
        for value in ordered_values[:-1]:
            left_dataset,right_dataset = filter_dataset(training_dataset,attribute,value)
            left_dataset_size, right_dataset_size = left_dataset.shape[0], right_dataset.shape[0]
            information_gain = entropy - left_dataset_size/dataset_size * calc_entropy(left_dataset) - right_dataset_size/dataset_size * calc_entropy(right_dataset)
            if information_gain > max_IG:
                max_IG = information_gain
                split_value = value
                split_attribute = attribute
    return split_attribute,split_value,max_IG

def filter_dataset(training_dataset,attribute,value):
    left_dataset_filter = training_dataset[:,attribute] <= value
    right_dataset_filter = training_dataset[:,attribute] > value
    left_dataset = training_dataset[left_dataset_filter]
    right_dataset = training_dataset[right_dataset_filter]
    return left_dataset,right_dataset

def calc_entropy(training_dataset):
    label_count = defaultdict(lambda: 0)
    dataset_size = training_dataset.shape[0]
    entropy = 0
    for label in training_dataset[:,-1]:
        label_count[label] += 1
    for value in label_count.values():
        ratio = value/dataset_size
        entropy -= ratio*np.log2(ratio)
    return entropy


# In[42]:


def count_node(node,count_dict):
    count_dict[node.depth]+=1
    if not node.is_a_leaf:
        count_node(node.left,count_dict)
        count_node(node.right,count_dict)
    return

def assign_position(node,count,track,max_width,max_depth):
    track[node.depth]+=1
    x = max_width*3.5/(count[node.depth]+1)*track[node.depth]-2
    y = node.depth*6+1.5
    node.position=(x,y)
    if node.depth == 0:
        node.p_position = (x,y)
    if not node.is_a_leaf:
        node.left.p_position=(x,y-0.05)
        node.right.p_position=(x,y-0.05)
        assign_position(node.left,count,track,max_width,max_depth)
        assign_position(node.right,count,track,max_width,max_depth)
    return

def add_boxes(nodeboxes,leafboxes,node):
    if node.is_a_leaf:
        leafboxes.append(Rectangle(node.position,3,1))
    else:
        nodeboxes.append(Rectangle(node.position,3,1))
        add_boxes(nodeboxes,leafboxes,node.left)
        add_boxes(nodeboxes,leafboxes,node.right)
    return

def print_text(node):
    if node.is_a_leaf:
        plt.text(node.position[0],node.position[1]+0.8, "value="+str(node.leaf_value), size = 35,                 family = "fantasy", color = "k", style = "italic", weight = "light")
    else:
        plt.text(node.position[0],node.position[1]+0.8, "A"+str(node.attribute)+'>'+str(node.value), size = 35,                 family = "fantasy", color = "k", style = "italic", weight = "light")
        plt.text((node.position[0]+node.left.position[0])/2+1.5,(node.position[1]+node.left.position[1])/2+0.8,'F',size=27,                 family = "fantasy", color = "k", style = "italic", weight = "light")
        plt.text((node.position[0]+node.right.position[0])/2+1.5,(node.position[1]+node.right.position[1])/2+0.8,'T',size=27,                 family = "fantasy", color = "k", style = "italic", weight = "light")
        print_text(node.left)
        print_text(node.right)
        return

def draw_line(node):
    if not node.is_a_leaf:
        plt.plot([node.position[0]+1.5,node.left.position[0]+1.5],[node.position[1]+1,node.left.position[1]])
        plt.plot([node.position[0]+1.5,node.right.position[0]+1.5],[node.position[1]+1,node.right.position[1]])
        draw_line(node.left)
        draw_line(node.right)
        
        
def make_node_boxes(ax, root,edgecolor='k', alpha=0.3):
    # Create list for all the error patches
    nodeboxes = []
    leafboxes = []
    add_boxes(nodeboxes,leafboxes,root)
    # Create patch collection with specified colour/alpha
    node_pc = PatchCollection(nodeboxes, facecolor='y', alpha=alpha, edgecolor=edgecolor)
    leaf_pc = PatchCollection(leafboxes, facecolor='g', alpha=alpha, edgecolor=edgecolor)
    # Add collection to axes
    ax.add_collection(node_pc)
    ax.add_collection(leaf_pc)
    print_text(root)
    draw_line(root)
    return

def draw(root,max_width,max_depth,filename):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(max_width*3.5,max_depth*3+1))
    ax.set_xlim(0,max_width*3.5)
    ax.set_ylim(max_depth*6+6)
    make_node_boxes(ax, root)
    plt.savefig(filename)
    plt.close(fig)
    plt.show()
    return

def draw_tree(root,depth,filename):
    depth_node_count = defaultdict(lambda: 0)
    depth_node_track = defaultdict(lambda: 0)
    count_node(root,depth_node_count)
    max_width = max(depth_node_count.values())
    max_depth = depth
    assign_position(root,depth_node_count,depth_node_track,max_width,max_depth)
    draw(root,max_width,max_depth,filename)
    return


# In[161]:


def forward(data,node):
    if node.is_a_leaf:
        return int(data[-1]),int(node.leaf_value)
    else:
        if data[node.attribute]>node.value:
            predict,actual = forward(data,node.right)
        else:
            predict,actual = forward(data,node.left)
        return predict,actual
    
  
def calc_recall(confusion_matrix,label):
    if np.sum(confusion_matrix[label]) == 0:
        return 0.0
    return confusion_matrix[label,label]/np.sum(confusion_matrix[label])


def calc_precision(confusion_matrix,label):
    if np.sum(confusion_matrix[:,label]) == 0:
        return 0.0
    return confusion_matrix[label,label]/np.sum(confusion_matrix[:,label])


def calc_F1(recall,precision):
    if recall == 0 or precision == 0:
        return 0.0
    return 2/(1/(recall+1)+1/precision)


def evaluate(test_db,trained_tree):
    confusion_matrix = np.zeros((4,4))
    for data in test_db:
        predict,actual = forward(data,trained_tree)
        confusion_matrix[actual-1,predict-1] += 1
    label_arr = np.zeros((4,3))
    for label in range(4):
        recall = calc_recall(confusion_matrix,label)
        precision = calc_precision(confusion_matrix,label)
        F1 = calc_F1(recall,precision)
        label_arr[label,0]=recall
        label_arr[label,1]=precision
        label_arr[label,2]=F1
    classification_rate = np.sum(confusion_matrix*np.identity(4))/np.sum(confusion_matrix)
    return confusion_matrix,label_arr,classification_rate


# In[162]:


def copy_tree(node):
    if node.is_a_leaf:
        return deepcopy(node)
    else:
        node_copy = deepcopy(node)
        node_copy.left = copy_tree(node.left)
        node_copy.right = copy_tree(node.right)
        return node_copy
    
    
def next_double_leaves_node(node):
    if node.is_a_leaf or node.keep:
        return None
    elif node.left.is_a_leaf and node.right.is_a_leaf and not node.keep:
        return node
    else:
        next_node = next_double_leaves_node(node.left)
        if not next_node:
            next_node = next_double_leaves_node(node.right)
        return next_node
    
    
def prun(root,dataset):
    root_copy = copy_tree(root)
    _,_,ori_acc = evaluate(dataset,root_copy)
    next_node = next_double_leaves_node(root_copy)
    while next_node:
        next_node.is_a_leaf = True
        next_node.leaf_value = next_node.major_value
        _,_,acc = evaluate(dataset,root_copy)
        if acc<=ori_acc:
            next_node.keep = True
            next_node.is_a_leaf = False
        else:
            ori_acc = acc
            next_node.left = None
            next_node.right = None
        next_node = next_double_leaves_node(root_copy)
    return root_copy


# In[173]:


def process(dataset_name,n=10,pruning=False,seed=0,draw=True):
    print('procissing',dataset_name,',pruning=',str(pruning),':')
    result_folder = dataset_name[:-4]+'_result'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    CMs = np.zeros((4,4))
    stat = np.zeros((4,3))
    average_classification_rate = 0
    average_depth = 0
    dataset = np.loadtxt(dataset_name)
    np.random.seed(seed)
    np.random.shuffle(dataset)
    for i in range(n):
        train_dataset,test_dataset = split_dataset(dataset,n,i)
        if pruning:
            best_root = None
            best_depth = None
            best_rate = 0.001
            for j in range(9):
                print(str(i+j/9)[:3]+'/10',end = '\r',flush = True)
                new_train_dataset,validationg_dataset = split_dataset(train_dataset,9,j)
                root,depth = decision_tree_learning(new_train_dataset,0)
                root = prun(root,validationg_dataset)
                img_name = 'plt'+str(i)+'_'+str(j)+'_prun.png'
                if draw:
                    draw_tree(root,depth,os.path.join(result_folder,img_name))
                confusion_matrix,label_arr,classification_rate = evaluate(validationg_dataset,root)
                if best_rate < classification_rate:
                    best_root = root
                    best_depth = depth 
                    best_rate = classification_rate
            confusion_matrix,label_arr,classification_rate = evaluate(test_dataset,best_root)
            CMs+=confusion_matrix
            stat += label_arr
            average_classification_rate += classification_rate
            average_depth += best_depth  
        else:
            print(str(i)+'/10',end = '\r',flush = True)
            root,depth = decision_tree_learning(train_dataset,0)
            img_name = 'plt_'+str(i)+'.png'
            if draw:
                draw_tree(root,depth,os.path.join(result_folder,img_name))
            confusion_matrix,label_arr,classification_rate = evaluate(test_dataset,root)
            CMs+=confusion_matrix
            stat += label_arr
            average_classification_rate += classification_rate
            average_depth += depth    

    CMs /= n
    stat /= n
    average_classification_rate /= n
    average_depth /= n
    print('ave--confusion_matrix:')
    print(CMs)
    print('ave--recall for each label:')
    print(stat[0,0],stat[1,0],stat[2,0])
    print('ave--precision for each label:')
    print(stat[0,1],stat[1,1],stat[2,1])
    print('ave--F1 score for each label:')
    print(stat[0,2],stat[1,2],stat[2,2])
    if pruning == False:
        print('average classification rate:',average_classification_rate)
        print('depth:',average_depth)
        print('-'*10)
    else:
        print('average classification rate:',average_classification_rate)
        print('depth:',average_depth)
        print('-'*10)


    return depth_list,cr_list,CMs,stat,average_classification_rate


# In[174]:


if __name__ == '__main__':
    np.seterr(invalid='ignore')
    clean_clean_CMs,clean_stat,clean_ave_class_rate = process(r'clean_dataset.txt',pruning=False,draw=False)
    clean_CMs_pruning,clean_stat_pruning,clean_ave_class_rate_pruning = process(r'clean_dataset.txt',pruning=True,draw=False)
    noisy_CMs,noisy_stat,noisy_ave_class_rate = process(r'noisy_dataset.txt',pruning=False,draw=False)
    noisy_CMs_pruning,noisy_stat_pruning,noisy_ave_class_rate_pruning = process(r'noisy_dataset.txt',pruning=True,draw=False)

