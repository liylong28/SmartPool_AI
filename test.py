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

        for speed_folder in os.listdir(self.data_root):


            if not speed_folder.startswith("dat_u"):
                continue


            V=float(
                speed_folder.replace(
                    "dat_u",
                    ""
                )
            )


            folder=os.path.join(
                self.data_root,
                speed_folder
            )


            for file in os.listdir(folder):


                if not file.endswith(".dat"):
                    continue


                H=float(
                    file.replace(
                        "suboff_h",
                        ""
                    ).replace(
                        ".dat",
                        ""
                    )
                )


                filepath=os.path.join(
                    folder,
                    file
                )


                X,Y,Z=self.read_tecplot(
                    filepath
                )


                self.samples.append(
                    {
                    "V":V,
                    "H":H,
                    "X":X,
                    "Y":Y,
                    "Z":Z
                    }
                )


        print(
            "Loaded samples:",
            len(self.samples)
        )



    def read_tecplot(
            self,
            filepath
    ):

        """
        读取Tecplot ASCII格式dat
        """

        with open(
            filepath,
            "r"
        ) as f:


            lines=f.readlines()



        data_start=None


        for i,line in enumerate(lines):


            if line.strip().startswith(
                "DT="
            ):

                data_start=i+1

                break



        if data_start is None:

            raise ValueError(
                "Cannot find data section"
            )



        data=[]


        for line in lines[data_start:]:


            line=line.strip()


            if line=="":

                continue


            values=line.split()


            if len(values)==3:

                data.append(
                    [
                    float(values[0]),
                    float(values[1]),
                    float(values[2])
                    ]
                )



        data=np.array(data)



        X=data[:,0]

        Y=data[:,1]

        Z=data[:,2]



        return X,Y,Z



    def __len__(self):

        return len(self.samples)



    def __getitem__(
            self,index
    ):

        return self.samples[index]
