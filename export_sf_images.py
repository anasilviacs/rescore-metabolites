"""
Script for exporting isotopic images as pickled files with numpy arrays
"""
# import cPickle
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import argparse
from os import path
from sm.engine.db import DB
from sm.engine.util import SMConfig, logger, proj_root

DS_CONFIG_SEL = "SELECT config, img_bounds FROM dataset WHERE name = %s"

EXPORT_SEL = ("SELECT f.sf, i.adduct, i.peak, i.pixel_inds, i.intensities "
              "FROM public.iso_image i, public.agg_formula f, public.dataset d "
              "WHERE i.sf_id = f.id AND i.job_id = d.id AND d.name = %s AND f.sf = %s AND array_length(i.pixel_inds, 1) > 0 "
              "ORDER BY i.adduct;")

def build_matrix(pxl_inds, ints, nrows, ncols):
   return coo_matrix((ints, (pxl_inds / ncols, pxl_inds % ncols)), shape=(nrows, ncols))


def get_img_dims(img_bounds):
    return (img_bounds['y']['max'] - img_bounds['y']['min'] + 1,
            img_bounds['x']['max'] - img_bounds['x']['min'] + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exporting isotopic images')
    parser.add_argument('ds_name', type=str, help='Dataset name')
    parser.add_argument('sf', type=str, help='sum formula')
    parser.add_argument('pkl_path', type=str, help='Path for the cPickle file')
    parser.add_argument('--config', dest='sm_config_path', type=str, help='SM config path')
    parser.set_defaults(sm_config_path=path.join(proj_root(), 'conf/config.json'))
    args = parser.parse_args()

    SMConfig.set_path(args.sm_config_path)
    db = DB(SMConfig.get_conf()['db'])

    ds_config, img_bounds = db.select_one(DS_CONFIG_SEL, args.ds_name)
    nrows, ncols = get_img_dims(img_bounds)
    isotope_gen_config = ds_config['isotope_generation']
    charge = '{}{}'.format(isotope_gen_config['charge']['polarity'], isotope_gen_config['charge']['n_charges'])
    export_rs = db.select(EXPORT_SEL, args.ds_name, args.sf)

    export_df = pd.DataFrame(export_rs, columns=['sf', 'adduct', 'peak', 'pxl_inds', 'ints'])
    export_df['img'] = export_df.apply(lambda r: build_matrix(np.array(r['pxl_inds']),
                                                              np.array(r['ints']), nrows, ncols), axis=1)
    export_df.drop(['pxl_inds', 'ints'], axis=1, inplace=True)
    # export_df.to_csv(args.csv_path, index=False)
    # cPickle.dump(export_df, open(args.pkl_path, 'wb'))
    export_df.to_csv(args.pkl_path)
    logger.info('Exported all images for "%s" sum formula in "%s" dataset into "%s" file', args.sf, args.ds_name, args.pkl_path)
