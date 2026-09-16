import os
import numpy as np


class WaveDataset:


    def __init__(
            self,
            data_root
    ):

        self.data_root=data_root

        self.samples=[]

        self.load()


    def load(self):

        """
        读取所有.dat文件
        """

        for speed_folder in os.listdir(
            self.data_root
        ):


            if not speed_folder.startswith(
                "dat_u"
            ):
                continue


            # 提取速度

            V=float(
                speed_folder.replace(
                    "dat_u",
                    ""
                )
            )


            speed_path=os.path.join(
                self.data_root,
                speed_folder
            )


            for file in os.listdir(speed_path):


                if not file.endswith(
                    ".dat"
                ):
                    continue


                # 提取深度

                H=float(
                    file.replace(
                        "suboff_h",
                        ""
                    )
                    .replace(
                        ".dat",
                        ""
                    )
                )


                filepath=os.path.join(
                    speed_path,
                    file
                )


                X,Y,Z=self.read_dat(
                    filepath
                )


                self.samples.append({

                    "V":V,

                    "H":H,

                    "X":X,

                    "Y":Y,

                    "Z":Z

                })


        print(
            f"Loaded {len(self.samples)} samples"
        )



    def read_dat(
            self,
            filepath
    ):


        data=np.loadtxt(
            filepath
        )


        X=data[:,0]

        Y=data[:,1]

        Z=data[:,2]


        return X,Y,Z



    def __len__(self):

        return len(self.samples)



    def __getitem__(self,index):

        return self.samples[index]
