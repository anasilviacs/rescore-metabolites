"""
Script for exporting search results into csv file
"""
import argparse
from os import path

from sm.engine.db import DB
from sm.engine.util import SMConfig, logger, proj_root

DS_CONFIG_SEL = "SELECT config FROM dataset WHERE name = %s"

EXPORT_SEL = ("SELECT sf_db.name, subst_ids::text::text, names, sf, m.adduct, ")

metrics = ['chaos', 'spatial', 'spectral', 'image_corr_01', 'image_corr_02', 'image_corr_03', 'image_corr_12',
           'image_corr_13', 'image_corr_23', 'snr', 'percent_0s', 'peak_int_diff_0', 'peak_int_diff_1',
           'peak_int_diff_2', 'peak_int_diff_3', 'quart_1', 'quart_2', 'quart_3', 'ratio_peak_01',
           'ratio_peak_02', 'ratio_peak_03', 'ratio_peak_12', 'ratio_peak_13', 'ratio_peak_23', 'percentile_10',
           'percentile_20', 'percentile_30', 'percentile_40', 'percentile_50', 'percentile_60', 'percentile_70',
           'percentile_80', 'percentile_90']

for feat in metrics:
    EXPORT_SEL = EXPORT_SEL + ("(m.stats->'{}')::text::real AS {}, ".format(feat, feat))

EXPORT_SEL = EXPORT_SEL + ("m.fdr::text::real, sigma, charge, pts_per_mz, tp.centr_mzs[1] "
              "FROM iso_image_metrics m "
              "JOIN formula_db sf_db ON sf_db.id = m.db_id "
              "JOIN agg_formula f ON f.id = m.sf_id AND sf_db.id = f.db_id "
              "JOIN job j ON j.id = m.job_id "
              "JOIN dataset ds ON ds.id = j.ds_id "
              "JOIN theor_peaks tp ON tp.db_id = sf_db.id AND tp.sf_id = m.sf_id AND tp.adduct = m.adduct "
              "WHERE sf_db.name = %s AND ds.name = %s "
              "AND ROUND(sigma::numeric, 6) = %s AND charge = %s AND pts_per_mz = %s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exporting search results into a tsv file')
    parser.add_argument('ds_name', type=str, help='Dataset name')
    parser.add_argument('csv_path', type=str, help='Path for the tsv file')
    parser.add_argument('--config', dest='sm_config_path', type=str, help='SM config path')
    parser.set_defaults(sm_config_path=path.join(proj_root(), 'conf/config.json'))
    args = parser.parse_args()

    SMConfig.set_path(args.sm_config_path)
    db = DB(SMConfig.get_conf()['db'])

    ds_config = db.select_one(DS_CONFIG_SEL, args.ds_name)[0]

    target_adducts = ds_config['isotope_generation']['adducts']

    isotope_gen_config = ds_config['isotope_generation']
    charge = '{}{}'.format(isotope_gen_config['charge']['polarity'], isotope_gen_config['charge']['n_charges'])
    export_rs = db.select(EXPORT_SEL, ds_config['database']['name'], args.ds_name,
                          isotope_gen_config['isocalc_sigma'], charge, isotope_gen_config['isocalc_pts_per_mz'])

    header = '\t'.join(['formula_db', 'db_ids', 'sf_name', 'sf', 'adduct']) +'\t' + '\t'.join(metrics) + '\t' + \
             '\t'.join(['fdr', 'isocalc_sigma', 'isocalc_charge', 'isocalc_pts_per_mz', 'first_peak_mz', 'targets']) + '\n'
    with open(args.csv_path, 'w') as f:
        f.write(header)
        f.writelines(['\t'.join(map(str, row)) + '\t' + str(target_adducts) + '\n' for row in export_rs])
    logger.info('Exported all search results for "%s" dataset into "%s" file', args.ds_name, args.csv_path)
