import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Route, Routes } from "react-router-dom";
import AboutUs from "./Page/AboutUs";
import Lipid from "./Page/Lipid";
import Upload from "./Page/Upload";
import { evaluateModel, setEvalData } from "./Slices/EvaluationSlice";

function App() {
  const data = useSelector((state) => state.evaluation.data);
  const dispatch = useDispatch()

  useEffect(() => {
    let dd = localStorage.getItem("data");
    // console.log(data)
    if (dd) {
      dd = JSON.parse(dd);
      dispatch(setEvalData(dd));
    } else {
      dispatch(evaluateModel())
    }
  }, []);

  return (
    <Routes>
      <Route path="/" element={<Lipid />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/about-us" element={<AboutUs />} />
    </Routes>
  );
}

export default App;
