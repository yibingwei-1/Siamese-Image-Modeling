import os
import torch.utils.data as data
import torchvision.datasets as datasets
import data_tools

__all__ = ['imagenet', 'imagenet100', 'cifar10', 'cifar100',
           'caltech101', 'oxford_pets', 'dtd', 'eurostat', 'oxford_flowers', 'stanford_cars', 'sun397', 'ucf101']


NUM_CLASSES = {
    'imagenet': 1000,
    'imagenet100': 100,
    'cifar10': 10,
    'cifar100': 100,
    'caltech101': 100,
    'oxford_pets': 101,
    'dtd': 47,
    'eurostat': 10,
    'oxford_flowers': 102,
    'stanford_cars': 196,
    'sun397': 397,
    'ucf101': 101,
}


class ImageListDataset(data.Dataset):
    def __init__(self, image_list, label_list, class_desc=None, transform=None):
        """TODO: to be defined.
        :pair_filelist: TODO
        """
        from torchvision.datasets.folder import default_loader
        self.loader = default_loader
        data.Dataset.__init__(self)
        self.samples = [(fn, lbl) for fn, lbl in zip(image_list, label_list)]
        self.transform = transform
        self.class_desc = class_desc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def imagenet(data_path, transform, train=True):
    import scipy.io
    meta = scipy.io.loadmat(os.path.join(data_path, 'anno', 'meta.mat'))['synsets']
    synsets = [m[0][1][0] for m in meta[:1000]]
    descriptions = {m[0][1][0]: m[0][2][0] for m in meta[:1000]}
    imagenet_ids = {m[0][1][0]: m[0][0][0][0]-1 for m in meta[:1000]}
    if train:
        dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform)
        wnids = [fn.split('/')[-2] for fn, lbl in dataset.samples]
        inids = [imagenet_ids[w] for w in wnids]

        dataset.samples = [(fn, lbl) for (fn, _), lbl in zip(dataset.samples, inids)]
        dataset.imgs = dataset.samples
        dataset.targets = inids
    else:
        fns = [f"{data_path}/val/ILSVRC2012_val_{i:08d}.JPEG" for i in range(1,50001)]
        inids = [int(ln.strip())-1 for ln in open(os.path.join(data_path, 'anno', 'ILSVRC2012_validation_ground_truth.txt'))]

        dataset = ImageListDataset(fns, inids, transform=transform)
        dataset.imgs = dataset.samples
        dataset.targets = inids

    dataset.classes = synsets
    dataset.descriptions = [descriptions[cls] for cls in dataset.classes]
    return dataset


def imagenet100(data_path, transform, train=True):
    imnet100_cls_list = ['n02869837','n01749939','n02488291','n02107142','n13037406','n02091831','n04517823','n04589890','n03062245','n01773797','n01735189','n07831146','n07753275','n03085013','n04485082','n02105505','n01983481','n02788148','n03530642','n04435653','n02086910','n02859443','n13040303','n03594734','n02085620','n02099849','n01558993','n04493381','n02109047','n04111531','n02877765','n04429376','n02009229','n01978455','n02106550','n01820546','n01692333','n07714571','n02974003','n02114855','n03785016','n03764736','n03775546','n02087046','n07836838','n04099969','n04592741','n03891251','n02701002','n03379051','n02259212','n07715103','n03947888','n04026417','n02326432','n03637318','n01980166','n02113799','n02086240','n03903868','n02483362','n04127249','n02089973','n03017168','n02093428','n02804414','n02396427','n04418357','n02172182','n01729322','n02113978','n03787032','n02089867','n02119022','n03777754','n04238763','n02231487','n03032252','n02138441','n02104029','n03837869','n03494278','n04136333','n03794056','n03492542','n02018207','n04067472','n03930630','n03584829','n02123045','n04229816','n02100583','n03642806','n04336792','n03259280','n02116738','n02108089','n03424325','n01855672','n02090622']

    def filter_dataset(db, cls_list):
        cls2lbl = {cls: lbl for lbl, cls in enumerate(cls_list)}
        idx = [i for i, lbl in enumerate(db.targets) if db.classes[lbl] in cls2lbl]
        samples = [db.samples[i] for i in idx]
        db.samples = [(fn, cls2lbl[db.classes[lbl]]) for fn, lbl in samples]
        db.targets = [dt[1] for dt in db.samples]
        db.imgs = db.samples
        descriptions = {cls: desc for cls, desc in zip(db.classes, db.descriptions)}
        db.classes = cls_list
        db.descriptions = [descriptions[cls] for cls in cls_list]

    dataset = imagenet(data_path, transform, train=train)
    filter_dataset(dataset, imnet100_cls_list)
    return dataset


def imagenet100_2(args, transform, train=True):
    from torchvision.datasets import ImageFolder
    return ImageFolder(root=args.data_path, transform=transform)


def cifar10(args, transform, train=True):
    if train:
        return datasets.CIFAR10(args.data_path, transform=transform, download=True, train=True)
    else:
        return datasets.CIFAR10(args.data_path, transform=transform, download=True, train=False)


def cifar100(args, transform, train=True):
    if train:
        return datasets.CIFAR100(args.data_path, transform=transform, download=True, train=True)
    else:
        return datasets.CIFAR100(args.data_path, transform=transform, download=True, train=False)


def caltech101(path, transform=None, num_shots=16, seed=None, train=True):
    dataset_dir = os.path.abspath(os.path.expanduser(path))
    image_dir = os.path.join(dataset_dir, "101_ObjectCategories")
    split_path = os.path.join(dataset_dir, "split_zhou_Caltech101.json")

    train_split, val_split, _ = data_tools.read_splits(split_path, image_dir)

    if num_shots >= 1:
        train_split = data_tools.generate_fewshot_dataset(train_split, num_shots=num_shots, seed=seed)
    db = train_split if train else val_split
    img_list = [dt[0] for dt in db]
    label_list = [dt[1] for dt in db]
    lbl2desc = {lbl: db[label_list.index(lbl)][2] for lbl in list(set(label_list))}
    return ImageListDataset(image_list=img_list, label_list=label_list, class_desc=lbl2desc, transform=transform)


def oxford_pets(path, transform=None, num_shots=16, seed=None, train=True):
    dataset_dir = os.path.abspath(os.path.expanduser(path))
    image_dir = os.path.join(dataset_dir, "images")
    split_path = os.path.join(dataset_dir, "split_zhou_OxfordPets.json")

    train_split, val_split, _ = data_tools.read_splits(split_path, image_dir)

    if num_shots >= 1:
        train_split = data_tools.generate_fewshot_dataset(train_split, num_shots=num_shots, seed=seed)

    db = train_split if train else val_split
    img_list = [dt[0] for dt in db]
    label_list = [dt[1] for dt in db]
    lbl2desc = {lbl: db[label_list.index(lbl)][2] for lbl in list(set(label_list))}
    return ImageListDataset(image_list=img_list, label_list=label_list, class_desc=lbl2desc, transform=transform)


def dtd(path, transform=None, num_shots=16, seed=None, train=True):
    dataset_dir = os.path.abspath(os.path.expanduser(path))
    image_dir = os.path.join(dataset_dir, "images")
    split_path = os.path.join(dataset_dir, "split_zhou_DescribableTextures.json")

    train_split, val_split, _ = data_tools.read_splits(split_path, image_dir)

    if num_shots >= 1:
        train_split = data_tools.generate_fewshot_dataset(train_split, num_shots=num_shots, seed=seed)

    db = train_split if train else val_split
    img_list = [dt[0] for dt in db]
    label_list = [dt[1] for dt in db]
    lbl2desc = {lbl: db[label_list.index(lbl)][2] for lbl in list(set(label_list))}
    return ImageListDataset(image_list=img_list, label_list=label_list, class_desc=lbl2desc, transform=transform)


def eurostat(path, transform=None, num_shots=16, seed=None, train=True):
    dataset_dir = os.path.abspath(os.path.expanduser(path))
    image_dir = os.path.join(dataset_dir, "2750")
    split_path = os.path.join(dataset_dir, "split_zhou_EuroSAT.json")

    train_split, val_split, _ = data_tools.read_splits(split_path, image_dir)

    if num_shots >= 1:
        train_split = data_tools.generate_fewshot_dataset(train_split, num_shots=num_shots, seed=seed)

    db = train_split if train else val_split
    img_list = [dt[0] for dt in db]
    label_list = [dt[1] for dt in db]
    lbl2desc = {lbl: db[label_list.index(lbl)][2] for lbl in list(set(label_list))}
    return ImageListDataset(image_list=img_list, label_list=label_list, class_desc=lbl2desc, transform=transform)


def oxford_flowers(path, transform=None, num_shots=16, seed=None, train=True):
    dataset_dir = os.path.abspath(os.path.expanduser(path))
    image_dir = os.path.join(dataset_dir, "jpg")
    split_path = os.path.join(dataset_dir, "split_zhou_OxfordFlowers.json")

    train_split, val_split, _ = data_tools.read_splits(split_path, image_dir)

    if num_shots >= 1:
        train_split = data_tools.generate_fewshot_dataset(train_split, num_shots=num_shots, seed=seed)

    db = train_split if train else val_split
    img_list = [dt[0] for dt in db]
    label_list = [dt[1] for dt in db]
    lbl2desc = {lbl: db[label_list.index(lbl)][2] for lbl in list(set(label_list))}
    return ImageListDataset(image_list=img_list, label_list=label_list, class_desc=lbl2desc, transform=transform)


def stanford_cars(path, transform=None, num_shots=16, seed=None, train=True):
    dataset_dir = os.path.abspath(os.path.expanduser(path))
    image_dir = os.path.join(dataset_dir, "jpg")
    split_path = os.path.join(dataset_dir, "split_zhou_StanfordCars.json")

    train_split, val_split, _ = data_tools.read_splits(split_path, image_dir)
    train_split = [(fn, lbl, ' '.join(desc.split()[1:]+desc.split()[:1])) for fn, lbl, desc in train_split]
    val_split = [(fn, lbl, ' '.join(desc.split()[1:]+desc.split()[:1])) for fn, lbl, desc in val_split]
    train_split = [(f"{dataset_dir}/train/{desc.replace('/', '-')}/{fn.split('/')[-1]}", lbl, desc) for fn, lbl, desc in train_split]
    val_split = [(f"{dataset_dir}/train/{desc.replace('/', '-')}/{fn.split('/')[-1]}", lbl, desc) for fn, lbl, desc in val_split]

    if num_shots >= 1:
        train_split = data_tools.generate_fewshot_dataset(train_split, num_shots=num_shots, seed=seed)

    db = train_split if train else val_split
    img_list = [dt[0] for dt in db]
    label_list = [dt[1] for dt in db]
    lbl2desc = {lbl: db[label_list.index(lbl)][2] for lbl in list(set(label_list))}
    return ImageListDataset(image_list=img_list, label_list=label_list, class_desc=lbl2desc, transform=transform)


def sun397(path, transform=None, num_shots=16, seed=None, train=True):
    dataset_dir = os.path.abspath(os.path.expanduser(path))
    image_dir = os.path.join(dataset_dir, "SUN397")
    split_path = os.path.join(dataset_dir, "split_zhou_SUN397.json")

    train_split, val_split, _ = data_tools.read_splits(split_path, image_dir)

    if num_shots >= 1:
        train_split = data_tools.generate_fewshot_dataset(train_split, num_shots=num_shots, seed=seed)

    db = train_split if train else val_split
    img_list = [dt[0] for dt in db]
    label_list = [dt[1] for dt in db]
    lbl2desc = {lbl: db[label_list.index(lbl)][2] for lbl in list(set(label_list))}
    return ImageListDataset(image_list=img_list, label_list=label_list, class_desc=lbl2desc, transform=transform)


def ucf101(path, transform=None, num_shots=16, seed=None, train=True):
    dataset_dir = os.path.abspath(os.path.expanduser(path))
    image_dir = os.path.join(dataset_dir, "UCF-101-midframes")
    split_path = os.path.join(dataset_dir, "split_zhou_UCF101.json")

    train_split, val_split, _ = data_tools.read_splits(split_path, image_dir)

    if num_shots >= 1:
        train_split = data_tools.generate_fewshot_dataset(train_split, num_shots=num_shots, seed=seed)

    db = train_split if train else val_split
    img_list = [dt[0] for dt in db]
    label_list = [dt[1] for dt in db]
    lbl2desc = {lbl: db[label_list.index(lbl)][2] for lbl in list(set(label_list))}
    return ImageListDataset(image_list=img_list, label_list=label_list, class_desc=lbl2desc, transform=transform)


def load_dataset(dataset, path, transform, train=True, **kwargs):
    if dataset == 'imagenet':
        return imagenet(path, transform, train)
    elif dataset == 'imagenet100':
        return imagenet100(path, transform, train)
    elif dataset == 'imagenet100-2':
        return imagenet100_2(path, transform, train)
    elif dataset == 'cifar10':
        return cifar10(path, transform, train)
    elif dataset == 'cifar100':
        return cifar100(path, transform, train)
    elif dataset == 'caltech101':
        return caltech101(path, transform=transform, train=train, **kwargs)
    elif dataset == 'oxford_pets':
        return oxford_pets(path, transform=transform, train=train, **kwargs)
    elif dataset == 'dtd':
        return dtd(path, transform=transform, train=train, **kwargs)
    elif dataset == 'eurostat':
        return eurostat(path, transform=transform, train=train, **kwargs)
    elif dataset == 'oxford_flowers':
        return oxford_flowers(path, transform=transform, train=train, **kwargs)
    elif dataset == 'stanford_cars':
        return stanford_cars(path, transform=transform, train=train, **kwargs)
    elif dataset == 'sun397':
        return sun397(path, transform=transform, train=train, **kwargs)
    elif dataset == 'ucf101':
        return ucf101(path, transform=transform, train=train, **kwargs)


from util import misc
from torch.utils import data
def create_loader(dataset, batch_size, num_workers=0, num_tasks=1, global_rank=0, pin_memory=True, drop_last=True):
    if misc.is_dist_avail_and_initialized():
        sampler_train = data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = data.RandomSampler(dataset)

    loader = data.DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader