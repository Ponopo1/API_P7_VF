import unittest
import requests
import joblib
import pandas as pd
from Prediction_api import predict_from_id_client, liste_client_df_api,liste_client_base_client

class TestApi(unittest.TestCase):
    
    # Test pour vérifier le chargement de la base de données x_test (df_api)
    def test_liste_client_df_api(self):
        # Given
        client_list = liste_client_df_api()
        # When
        computed_result = len(set(client_list))
        # Then 
        assert computed_result == 92253 # Nb à changer si changement de données
        print('Nous avons chargé le bon fichier X_test')

    # Test pour vérifier si le prédict fonctionne bien
    def test_predict_from_id_client_when_id_client_does_not_exist(self):
        # Given 
        non_existent_id_client = 10000000000
        # When        
        computed_result = predict_from_id_client(non_existent_id_client)
        # Then
        assert computed_result is None
        print('Le faux client n\'existe pas')

    def test_predict_from_id_client_when_id_client_does_exist(self):
        # Given 
        existent_id_client = 405321
        # When        
        computed_result = predict_from_id_client(existent_id_client)
        # Then
        expected_result = 0.16238979859806946
        assert expected_result-computed_result < 0.001
        print(f"Le resultat prédict pour {existent_id_client} est bon") 
    
    # Test de nos données générales
    def test_info_client_when_id_does_not_exist(self):
        # Given 
        non_existent_id_client = 10000000000
        # When        
        computed_result = liste_client_base_client(non_existent_id_client)
        # Then
        assert computed_result is None
        print('Pas de fausse donnée chargée')
    
    def test_info_client_when_id_does_not_exist(self):
        # Given 
        existent_id_client = 405321
        # When        
        computed_result = liste_client_base_client(existent_id_client)
        # Then
        expected_result = {
            'AMT_CREDIT': {405321: 247500.0},
            'AMT_ANNUITY': {405321: 9814.5},
            'AMT_GOODS_PRICE': {405321: 247500.0},
            'DAYS_BIRTH': {405321: -10916},
            'CNT_CHILDREN': {405321: 0},
            'DAYS_EMPLOYED': {405321: -810.0},
            'NAME_EDUCATION_TYPE': {405321: 'Higher education'},
            'NAME_CONTRACT_TYPE': {405321: 'Cash loans'},
            'NAME_FAMILY_STATUS': {405321: 'Married'},
            'NAME_HOUSING_TYPE': {405321: 'House / apartment'},
            'NAME_INCOME_TYPE': {405321: 'State servant'},
            'CODE_GENDER': {405321: 'F'}
        }
        self.assertEqual(computed_result, expected_result)
        print(f"Le resultat général pour {existent_id_client} est correct")
 
if __name__ == '__main__':
    unittest.main()

