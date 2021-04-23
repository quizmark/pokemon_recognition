#!/bin/bash
DOWNLOAD_LINK="https://storage.googleapis.com/kaggle-data-sets/40205/63131/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1589167723&Signature=oh7ynAHtl8GUdkGMDavVu%2FEmGf6KVJgR%2FQ1tN2KinDs2vCEfV9WsYYa1UfdB%2FIWmUXHVbVR9g%2FBbt3%2BAUHuSE%2F6qno960BusvihyX8ZXzKjJa5HiibtAGEaIVqtg6cxajB0oUEzVFewle8LO9tMRLsPTGA1VnswtiVUl0B%2B2ZIc1JSbYIK8%2BR5IrmSl8Holo1IOAfy1lu2bdqfVN9iAYFNpAEkf8cWocZZthmVoKm3mLi0aeTlb40A5%2FbzXhcHtc1iCNPiQn%2F7bA1MvJQOt%2Bn0TSIkVHvPwpbWehsXkuEpqaEwBNn43wO3pKee%2BKxLYqcsHWqlll6a2A9Q%2BzWXDVXw%3D%3D&response-content-disposition=attachment%3B+filename%3Dpokemon-generation-one.zip"

DATASET_SOURCE="https://www.kaggle.com/thedagger/pokemon-generation-one"

DATASET_ZIP="dataset.zip"

{
	{
		unzip $DATASET_ZIP
	} || {
		wget $DOWNLOAD_LINK && mv archive.zip?GoogleAccessId* $DATASET_ZIP && unzip $DATASET_ZIP
}
} || {
	echo "Please download dataset from \`$DATASET_SOURCE\` and rename into \`$DATASET_ZIP\`"
}

rm -rf dataset/dataset
mkdir 'dataprep'
mkdir 'dataprep/train'
mkdir 'dataprep/test'
