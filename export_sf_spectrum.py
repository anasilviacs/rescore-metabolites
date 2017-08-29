"""
Script for exporting isotopic spectrum
"""
import numpy as np
import pandas as pd
import argparse
from os import path
from sm.engine.db import DB
from sm.engine.util import SMConfig, logger, proj_root

DS_CONFIG_SEL = "SELECT config, img_bounds FROM dataset WHERE name = %s"

EXPORT_SEL = ("SELECT f.sf, t.adduct, t.centr_mzs, t.centr_ints "
              "FROM public.agg_formula f, public.theor_peaks t "
              "WHERE t.sf_id = f.id AND f.sf = %s "
              "ORDER BY t.adduct;")


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
    export_df['img_dims'] = [(img_bounds['y']['min'], img_bounds['y']['max'], img_bounds['x']['min'], img_bounds['x']['max'])] * len(export_df)
    # export_df['img'] = export_df.apply(lambda r: build_matrix(np.array(r['pxl_inds']),
    #                                                           np.array(r['ints']), nrows, ncols), axis=1)
    # export_df.drop(['pxl_inds', 'ints'], axis=1, inplace=True)
    # export_df.to_csv(args.csv_path, index=False)
    # cPickle.dump(export_df, open(args.pkl_path, 'wb'))
    export_df.to_csv(args.pkl_path, index=False)
    logger.info('Exported all images for "%s" sum formula in "%s" dataset into "%s" file', args.sf, args.ds_name, args.pkl_path)
