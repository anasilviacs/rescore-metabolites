"""
Script for exporting isotopic spectrum
"""
import numpy as np
import pandas as pd
import argparse
from os import path
from sm.engine.db import DB
from sm.engine.util import SMConfig, logger, proj_root

EXPORT_SEL = ("SELECT f.sf, t.adduct, t.centr_mzs, t.centr_ints "
              "FROM public.agg_formula f, public.theor_peaks t "
              "WHERE t.sf_id = f.id AND f.db_id = 1 AND f.sf = %s AND t.adduct = %s " # hardcoded to always fetch from HMDB, lazy i know
              "ORDER BY t.adduct;")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exporting isotopic images')
    parser.add_argument('sf', type=str, help='sum formula')
    parser.add_argument('add', type=str, help='adduct')
    parser.add_argument('pkl_path', type=str, help='Path for the cPickle file')
    parser.add_argument('--config', dest='sm_config_path', type=str, help='SM config path')
    parser.set_defaults(sm_config_path=path.join(proj_root(), 'conf/config.json'))
    args = parser.parse_args()

    SMConfig.set_path(args.sm_config_path)
    db = DB(SMConfig.get_conf()['db'])

    export_rs = db.select(EXPORT_SEL, args.sf, args.add)

    export_df = pd.DataFrame(export_rs, columns=['sf', 'adduct', 'centr_mzs', 'centr_ints'])

    export_df.to_csv(args.pkl_path, index=False)
    logger.info('Exported the spectra for the "%s" sum formula, "%s" adduct into "%s" file', args.sf, args.add, args.pkl_path)
