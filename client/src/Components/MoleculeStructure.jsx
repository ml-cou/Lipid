import CheckIcon from "@mui/icons-material/Check";
import RemoveCircleOutlineIcon from "@mui/icons-material/RemoveCircleOutline";
import { CircularProgress } from "@mui/material";
import { Input } from "antd";
import React, { useEffect, useState } from "react";
import ChartComponent from "./ChartComponent";
import toast from "react-hot-toast";

function MoleculeStructure() {
  const [mol_name, setMolName] = useState("");
  const [molecules, setMolecules] = useState([]);
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(false);
  const [graph_data, setGraphData] = useState([]);

  useEffect(() => {
    setGraphData(Object.keys(data).map((val) => data[val]));
  }, [data]);

  const handleAddMolecule = async () => {
    const tempMolecules = [...molecules, { mol_name }];

    const wantToReq = tempMolecules.filter((val) => !data[val.mol_name]);
    // console.log(wantToReq);
    const promises = wantToReq.map((molecule) =>
      fetch("http://localhost:8000/edge_pred/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ mol_name: molecule.mol_name }),
      })
    );
    setLoading(true);
    try {
      const response = await Promise.all(promises);
      let d = await Promise.all(response.map((res) => res.json()));
      const receivedStrutctures = {};
      d.forEach((val) => {
        receivedStrutctures[val.predicted_edge.name] = val.predicted_edge;
      });
      setData({ ...data, ...receivedStrutctures });

      setMolName("");
      setMolecules(tempMolecules);
    } catch (error) {
      toast.error(error.message);
    }
    setLoading(false);
  };

  return (
    <div className="p-4 w-full h-full grid grid-cols-12 gap-4">
      <div className="col-span-3 border-r shadow-r-lg pr-4">
        <div className="flex items-center gap-1.5">
          <Input
            size="large"
            value={mol_name}
            onChange={(e) => setMolName(e.target.value)}
          />
          <button
            className="px-2 h-10 flex items-center border rounded bg-blue-500 text-white"
            onClick={handleAddMolecule}
          >
            <CheckIcon className="!w-5 !h-5" />
          </button>
        </div>
        <div className="mt-4">
          {molecules.length || loading ? (
            molecules.map((val, ind) => (
              <div
                key={ind}
                className="flex items-center justify-between hover:bg-gray-100 px-2 py-1.5"
              >
                <p className="text-xl font-medium text-gray-700">
                  {ind + 1}. {val.mol_name}
                </p>
                <button
                  className="text-gray-600 hover:text-red-500"
                  onClick={() => {
                    const name = val.mol_name;
                    setMolecules(
                      molecules.filter((val) => val.mol_name !== name)
                    );

                    setData((prevData) => {
                      const newData = { ...prevData }; // Create a shallow copy of the data
                      delete newData[name]; // Delete the property
                      return newData; // Return the new object to update the state
                    });
                  }}
                >
                  <RemoveCircleOutlineIcon />
                </button>
              </div>
            ))
          ) : (
            <p className="text-center text-xl font-medium mt-12">
              Add new molecule to start...
            </p>
          )}
          {loading && (
            <div className="flex justify-center mt-4">
              <CircularProgress />
            </div>
          )}
        </div>
      </div>
      <div className="w-full h-full col-span-9">
        <ChartComponent graph_data={graph_data} />
      </div>
    </div>
  );
}


// Old Code

// function MoleculeStructure() {
//   const [type, setType] = useState("single");
//   const [lipidInput, setLipidInput] = useState([{ name: "", percentage: 100 }]);
//   const dispatch = useDispatch();
//   const { loading, data, showTable, lipid } = useSelector((state) => state.structure);
//   const [open, setOpen] = useState(false);

//   const handleInputChange = (index, field, value) => {
//     // Deep copy the object to ensure we're not modifying a read-only reference
//     const updatedInputs = lipidInput.map((input, idx) =>
//       idx === index ? { ...input, [field]: value } : { ...input }
//     );

//     // Update the state with the new array
//     setLipidInput(updatedInputs);
//     dispatch(changeActiveLipid(updatedInputs));
//   };

//   return (
//     <div className="w-full h-full">
//       <div
//         className={`mt-3 relative max-w-[350px] mx-auto border p-4 shadow rounded `}
//       >
//         <SelectCompositionType
//           setLipidInput={setLipidInput}
//           setType={setType}
//           type={type}
//         />
//         <LipidInputForm
//           handleInputChange={handleInputChange}
//           lipidInput={lipidInput}
//           type={type}
//         />

//         <button
//           className="mt-2 ml-auto  bg-blue-500 w-full py-2 text-white rounded"
//           onClick={() => {
//             if (type === "single") {
//               dispatch(getMoleculeStructure(lipidInput[0].name));
//             } else {
//               // TODO: Handle por multiple composition
//             }
//           }}
//         >
//           Analyze
//         </button>
//         {/* <span
//           className="border cursor-pointer p-1 px-1.5 absolute bottom-0 left-1/2 translate-y-2/3 -translate-x-1/2 bg-slate-100 shadow text-sm"
//           onClick={() => setCollapse(true)}
//         >
//           <KeyboardDoubleArrowDownIcon className="!w-5 !h-5 rotate-180" />
//         </span> */}
//       </div>

//       {loading ? (
//         <div className="flex flex-col items-center justify-center mt-10">
//           <h3 className="font-medium mb-2">Fetching Data...</h3>
//           <CircularProgress />
//         </div>
//       ) : (
//         <div className={`${data && "w-full h-full"}`}>
//           {data && (
//             <>
//               <h1 className="text-2xl font-bold font-mono text-center mt-8">
//                 Structure Analysis for {lipid[0].name}
//               </h1>
//               <div className="w-full h-full border p-2 mt-6 shadow rounded">
//                 <div className="text-center space-x-4 mt-2">
//                   <button
//                     className="p-2 bg-violet-500 shadow px-3 rounded text-sm text-white"
//                     onClick={() => dispatch(changeShowTable())}
//                   >
//                     {showTable ? "Hide" : "Show"} Table
//                   </button>
//                   <button
//                     className="p-2 bg-violet-500 shadow px-3 rounded text-sm text-white"
//                     onClick={() => setOpen(true)}
//                   >
//                     Actual vs Predicted
//                   </button>
//                 </div>
//                 <ActualPredicted open={open} setIsOpen={setOpen} />
//                 <div className="w-full h-full flex items-center">
//                   <ChartComponent
//                     id={Date.now()}
//                     graph_data={
//                       data &&
//                       data.predicted && [data.predicted[lipid[0].name]]
//                     }
//                   />
//                   <div className={`${showTable ? "w-[600px] px-2" : "w-0"}`}>
//                     <GraphTable />
//                   </div>
//                 </div>
//               </div>
//             </>
//           )}
//         </div>
//       )}
//     </div>
//   );
// }

export default MoleculeStructure;
