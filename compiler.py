import os
import json
import shutil
import random
import importlib
import cv2
import argparse
from tqdm.auto import tqdm


def is_image(name):
    """
    Checks is given file an image
    :param name: filename
    :return: True is file is image
    """
    extension = name.split('.')[-1]
    if extension.lower() in ['jpg', 'png', 'bmp', 'jpeg']:
        return True
    return False


def build_augmentations(source):
    """
    Creates list of augmentations classes.

    :param source: name of augmentation
    :return: list of augs
    """
    aug_names = source['augmentations']
    augs = []
    for aug in aug_names:
        AugClass = getattr(importlib.import_module(f'augmentations.aug_{aug}'), 'Augmentations')
        augs.append(AugClass())
    return augs


class Compiler:
    def __init__(self, cfg_path, use_debug, use_symlink):
        self.cfg_path = cfg_path
        self.use_debug = use_debug
        self.use_symlink = use_symlink
        
    def debug(self, string):
        if self.use_debug:
            print(string)
            
    def make_backup_if_needed(self, cfg):
        target_dir = cfg['target_dir']
        target_path = os.path.join(cfg['target_dir'], cfg['name'])
        if os.path.exists(target_path):
            self.debug(f'Some directory already exists with name: {cfg["name"]}')
            self.debug(f'Moving to: {target_path}_DDCbackup ...')
            backup_path = os.path.join(target_dir, cfg['name'] + '_DDCbackup')
            if os.path.exists(backup_path):
                try:
                    shutil.rmtree(backup_path)
                except OSError:
                    os.remove(backup_path)
            shutil.move(target_path, backup_path)
            self.debug(f'Successfully done.')

    def build_predefined_split(self, source, new_symlink_path, augs):
        self.debug(f'Taking train.txt, val.txt and test.txt from cfg...')
        with open(source['train'], 'r') as f:
            train_data = list(filter(lambda x: len(x) > 0, f.read().split('\n')))
        train_data = [os.path.join(os.path.abspath(new_symlink_path), os.path.basename(t)) for t in train_data]
        with open(source['valid'], 'r') as f:
            val_data = list(filter(lambda x: len(x) > 0, f.read().split('\n')))
        val_data = [os.path.join(os.path.abspath(new_symlink_path), os.path.basename(t)) for t in val_data]
        if 'test' in source:
            with open(source['test'], 'r') as f:
                test_data = list(filter(lambda x: len(x) > 0, f.read().split('\n')))
            test_data = [os.path.join(os.path.abspath(new_symlink_path), os.path.basename(t)) for t in test_data]
        else:
            test_data = val_data

        if augs is not None:
            print('https://github.com/VolkovAK/DarknetDatasetCompiler')
            raise NotImplemented('If you need augmentations with predefined split - create an issue on github, please')
        return train_data, val_data, test_data

    def build_random_split(self, source, new_symlink_path, augs):
        original_path = source['path']
        random.seed(42)
        train_part, val_part, test_part = source['split']
        self.debug(f'Scanning {original_path}...')
        imgs = [j for j in os.listdir(original_path) if is_image(j)]
        train_data = []
        val_data = []
        test_data = []
        self.debug(f'Splitting {original_path}...')
        random.shuffle(imgs)
        if 'use_part' in source:
            imgs = imgs[:int(source['use_part'] / 100 * len(imgs))]
        random.seed(42)

        for img_name in tqdm(imgs):

            txt_name = os.path.splitext(img_name)[0] + '.txt'
            txt_file = os.path.join(original_path, txt_name)
            img_file = os.path.join(original_path, img_name)
            if os.path.exists(txt_file) and os.path.exists(img_file):
                phase = random.randint(0, 100)
                if phase < train_part:
                    train_data.append(os.path.join(os.path.abspath(new_symlink_path), img_name))
                elif phase < train_part + val_part:
                    val_data.append(os.path.join(os.path.abspath(new_symlink_path), img_name))
                else:
                    test_data.append(os.path.join(os.path.abspath(new_symlink_path), img_name))

                random_state = random.getstate()
                if augs is not None:
                    img = cv2.imread(img_file)
                    for aug in augs:
                        img = aug.do(img)
                    cv2.imwrite(os.path.join(new_symlink_path, img_name), img)
                    shutil.copy(txt_file, os.path.join(new_symlink_path, txt_name))
                if augs is None and self.use_symlink is False:
                    shutil.copy(img_file, os.path.join(new_symlink_path, img_name))
                    shutil.copy(txt_file, os.path.join(new_symlink_path, txt_name))
                random.setstate(random_state)

        return train_data, val_data, test_data


    def run(self):
        cfg = json.load(open(self.cfg_path, 'r'))

        # save old dataset to backup directory
        self.make_backup_if_needed(cfg)

        target_path = os.path.join(cfg['target_dir'], cfg['name'])

        os.makedirs(target_path)
        train_txt = open(os.path.join(target_path, 'train.txt'), 'w')
        val_txt = open(os.path.join(target_path, 'val.txt'), 'w')
        test_txt = open(os.path.join(target_path, 'test.txt'), 'w')
        readme_txt = open(os.path.join(target_path, 'readme.txt'), 'w')
        total_train = 0
        total_val = 0
        total_test = 0
        
        for source in cfg['sources']:
            if source['use'] is False:
                self.debug(f'Source {source["name"]} will not be used, skip')
                continue
            original_path = source['path']
            name = source['name']
            new_symlink_path = os.path.join(target_path, name)
            multiplier = int(source['multiplier']) if 'multiplier' in source else 1
            self.debug(f'\nSource: {original_path}')
            self.debug(f'Creating dataset folder: {name}...')
            # for readme there is more sense in a new name, not in the actual data path
            readme_txt.write(f'Source: {name}\n')
            if 'description' in source:
                readme_txt.write(f'Description: {source["description"]}\n')

            augs = None
            if 'augmentations' in source:
                augs = build_augmentations(source)

            if self.use_symlink and augs is None:
                os.symlink(original_path, new_symlink_path)
            else:
                os.makedirs(new_symlink_path, exist_ok=True)

            if 'train' in source and 'valid' in source:
                train_data, val_data, test_data= self.build_predefined_split(source, new_symlink_path, augs)
            elif 'split' in source:
                train_data, val_data, test_data = self.build_random_split(source, new_symlink_path, augs)
            else:
                raise Exception('You should specify Split parameter of Train and Val paths')
            train_data = train_data * multiplier
        
            for d in train_data:
                train_txt.write(d + '\n')
            for d in val_data:
                val_txt.write(d + '\n')
            for d in test_data:
                test_txt.write(d + '\n')
            readme_txt.write(f'Train: {len(train_data)}\n')
            readme_txt.write(f'Valid: {len(val_data)}\n')
            readme_txt.write(f'Test: {len(test_data)}\n')
            if multiplier != 1:
                readme_txt.write(f'Multiplier: x{multiplier}\n')
            readme_txt.write('\n')
            self.debug(f'Train: {len(train_data)}')
            self.debug(f'Valid: {len(val_data)}')
            self.debug(f'Test: {len(test_data)}')

            total_train += len(train_data)
            total_val += len(val_data)
            total_test += len(test_data)

        train_txt.close()
        val_txt.close()
        test_txt.close()
        readme_txt.close()
        with open(os.path.join(target_path, 'readme.txt'), 'r') as f:
            readme_data = f.read()
        with open(os.path.join(target_path, 'readme.txt'), 'w') as readme_txt:
            readme_txt.write(f'Dataset: {cfg["name"]}\nTOTAL:\n')
            readme_txt.write(f'Train: {round(total_train/1000)}k\n')
            readme_txt.write(f'Valid: {round(total_val/1000)}k\n')
            readme_txt.write(f'Test: {round(total_test/1000)}k\n\n')
            readme_txt.write(readme_data)
        

def main():
    parser = argparse.ArgumentParser(description='Darknet Dataset Compiler')
    parser.add_argument('cfg_path', help='Path to configuration json file')
    parser.add_argument('-q', '--quiet', help='Quiet compilation', action='store_true')
    parser.add_argument('-c', '--copy', help='Copy data instead of symlink', action='store_true')
    args = parser.parse_args()

    use_debug = False if args.quiet is True else True
    use_symlink = False if args.copy is True else True
    cfg_path = args.cfg_path

    compiler = Compiler(cfg_path, use_debug, use_symlink)
    compiler.run()


if __name__ == '__main__':
    main()
