"""
Script for exporting search results into csv file
"""
import argparse
from os import path

from sm.engine.db import DB
from sm.engine.util import SMConfig, logger, proj_root

# EXPORT_SEL = ('SELECT t.sf_id, t.target_add, a.sf, t.decoy_add '
#               'FROM target_decoy_add t '
#               'JOIN agg_formula a on a.id = t.sf_id '
#               'JOIN dataset d ON d.id = t.job_id  '
#               'WHERE d.name = %s '
#               'ORDER BY t.target_add, t.sf_id')

EXPORT_SEL = ('SELECT adds.sf_id, adds.target_add, f.sf, adds.decoy_add '
              'FROM target_decoy_add adds '
              'JOIN agg_formula f ON f.id = adds.sf_id '
              'JOIN job j ON j.id = adds.job_id '
              'JOIN dataset ds ON ds.id = j.ds_id AND adds.db_id = f.db_id'
              'WHERE ds.name = %s '
              'ORDER BY adds.target_add, adds.sf_id')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exporting target/decoy sets into a csv file')
    parser.add_argument('ds_name', type=str, help='Dataset name')
    parser.add_argument('csv_path', type=str, help='Path for the csv file')
    parser.add_argument('--config', dest='sm_config_path', type=str, help='SM config path')
    parser.set_defaults(sm_config_path=path.join(proj_root(), 'conf/config.json'))
    args = parser.parse_args()

    SMConfig.set_path(args.sm_config_path)
    db = DB(SMConfig.get_conf()['db'])

    export_rs = db.select(EXPORT_SEL, args.ds_name)

    header = ','.join(['sf_id', 'target_add', 'sf', 'decoy_add']) + '\n'
    with open(args.csv_path, 'w') as f:
        f.write(header)
        f.writelines([','.join(map(str, row)) + '\n' for row in export_rs])

    logger.info('Exported all search results for "%s" dataset into "%s" file', args.ds_name, args.csv_path)
