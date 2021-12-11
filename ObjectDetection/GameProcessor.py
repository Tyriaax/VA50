import cv2.cv2

from homography import *
from pawns_recognition import *
from cards_recognition import *
from GameBoard import *
from drawing import *

class GameProcessor:
    capEveryFrame = False
    homographymatrixfound = False

    def __init__(self, img, window_name):
        self.window_name = window_name

        self.height = img.shape[0]
        self.width = img.shape[1]

        self.gameBoard = GameBoard()
        self.pawnsRecognitionHelper = PawnsRecognitionHelper(self.height, self.width, self.gameBoard)
        self.cardsRecognitionHelper = CardsRecognitionHelper(self.height, self.width, self.gameBoard)

        cv2.setMouseCallback(self.window_name, self.ComputeMouseInput)

        self.list_board_coords = []

    def ComputeFrame(self, img):
        if not self.homographymatrixfound:
            if len(self.list_board_coords) == 4:
                self.homographymatrix, self.coordinates = get_homography_matrix(img, np.array(self.list_board_coords), self.width, self.height)

                self.cardsRecognitionHelper.GetScreenPortions(img, self.coordinates)
                self.pawnsRecognitionHelper.GetScreenPortion(img, self.coordinates)
                self.homographymatrixfound = True
                img = cv2.warpPerspective(img, self.homographymatrix, (img.shape[1], img.shape[0]))
                if(self.gameBoard.tryUpdateGameStatus(GameStates.GSWaitingActionPawns)):
                    self.cardsRecognitionHelper.ComputeCards(img)
                    self.cardsRecognitionHelper.ComputeFrame(img)          
        else:
            img = cv2.warpPerspective(img, self.homographymatrix, (img.shape[1], img.shape[0]))

        return img

    def DrawFrame(self, img):
        modifiedimg = img.copy()

        if len(self.list_board_coords) < 4:
            for coord in self.list_board_coords:
                cv2.circle(modifiedimg, coord, 10, (0, 255, 0), -1)

        if self.gameBoard.getGameStatus() == GameStates.GSWaitingHomography:
            modifiedimg = drawText(modifiedimg, "Selectionnez les quatre coins des 9 cartes",TextPositions.TPCenter)
        if self.gameBoard.getGameStatus() == GameStates.GSWaitingActionPawns:
            modifiedimg = self.pawnsRecognitionHelper.DrawZonesRectangles(modifiedimg, drawOffset=True)
            modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = drawText(modifiedimg, "Appuyez sur P pour detecter les pions", TextPositions.TPTopL)
            modifiedimg = drawText(modifiedimg, "Ou sur C pour redetecter les cartes", TextPositions.TPTopL, 1)
        if self.gameBoard.getGameStatus() == GameStates.GSUseActionsPawns:
            modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = self.pawnsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = drawText(modifiedimg, "Realisez votre Action puis", TextPositions.TPTopL)
            modifiedimg = drawText(modifiedimg, "Appuyez sur le jeton correspondant", TextPositions.TPTopL, 1)
            modifiedimg = drawText(modifiedimg, "Ou sur P pour redetecter les pions", TextPositions.TPTopL, 2)
            modifiedimg = drawText(modifiedimg, "Tour : " + str(self.gameBoard.getTurnCount()) + "/" + str(self.gameBoard.getMaxTurnCount()), TextPositions.TPTopR)
            modifiedimg = drawText(modifiedimg, "Joueur : " + str(self.gameBoard.getCurrentPlayer()), TextPositions.TPTopR,1)
        if self.gameBoard.getGameStatus() == GameStates.GSAppealOfWitness:
            modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = self.pawnsRecognitionHelper.DrawFrame(modifiedimg)
            modifiedimg = drawText(modifiedimg, "Retournez les cartes innocentees", TextPositions.TPTopL)
            modifiedimg = drawText(modifiedimg, "Puis appuyez sur c", TextPositions.TPTopL, 1)
            modifiedimg = drawText(modifiedimg, "Tour : " + str(self.gameBoard.getTurnCount()) + "/" + str(self.gameBoard.getMaxTurnCount()), TextPositions.TPTopR)
        if self.gameBoard.getGameStatus() == GameStates.GSGameOver:
            modifiedimg = drawText(modifiedimg, "Partie terminee", TextPositions.TPCenter)
            if(self.gameBoard.getDetectiveWins()):
                modifiedimg = drawText(modifiedimg, "Vous avez gagne", TextPositions.TPCenter,1)
            else:
                modifiedimg = drawText(modifiedimg, "Jack a gagne", TextPositions.TPCenter,1)

        return modifiedimg

    def ComputeInputs(self, img):
        self.lastimg = img
        continuebool = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            continuebool = False
        if (key == ord('c') or self.capEveryFrame):
            if (self.gameBoard.getDetectiveWins() or self.gameBoard.getJackWins()):
                self.gameBoard.tryUpdateGameStatus(GameStates.GSGameOver)
            elif (self.gameBoard.tryUpdateGameStatus(GameStates.GSWaitingActionPawns)):
                self.cardsRecognitionHelper.ComputeCards(img)
                self.cardsRecognitionHelper.ComputeFrame(img)

        if (key == ord('p') or self.capEveryFrame):
            if (self.gameBoard.tryUpdateGameStatus(GameStates.GSUseActionsPawns)):
                self.pawnsRecognitionHelper.ComputeFrame(img)
                self.gameBoard.printState()
                self.gameBoard.manhunt()

        return continuebool

    def ComputeMouseInput(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.list_board_coords) < 4:
                self.list_board_coords.append([x, y])
            else:
                actionPawnIndex = self.pawnsRecognitionHelper.actionPawnClick([x,y])
                if actionPawnIndex is not None:
                    actionPawnClicked = self.gameBoard.getActionPawns()[actionPawnIndex]
                    selectedAP = ActionPawns[actionPawnClicked]
                    print("Action Pawn Clicked : " + actionPawnClicked)
                    if(selectedAP.value <= 4):
                        self.pawnsRecognitionHelper.ComputeDetectivePawns(self.lastimg)
                        print("Previous : \n", self.gameBoard.getPreviousDetectivePawns(), "\nCurrent: \n", self.gameBoard.getDetectivePawns())
                    elif(selectedAP.value <= 7):
                        self.cardsRecognitionHelper.ComputeCards(self.lastimg)
                        #print("Previous : \n", self.gameBoard.getPreviousCardsState(), "\nCurrent: \n", self.gameBoard.getCardsState())
                        self.cardsRecognitionHelper.ComputeFrame(self.lastimg)

                    if(self.gameBoard.IsActionPawnRespected(actionPawnClicked)):
                        print("Action Pawn Used")
                        self.pawnsRecognitionHelper.actionPawnUsed(actionPawnIndex)
                        if(selectedAP.value <= 4):
                            self.gameBoard.updatePreviousPawnsState()
                        else:
                            self.gameBoard.updatePreviousCardsState()
                        self.gameBoard.getNextPlayerToUseActionsPawns()
                        if(len(self.gameBoard.getActionPawns()) == 0):
                            print("Turn Finished")
                            if(self.gameBoard.tryUpdateGameStatus(GameStates.GSAppealOfWitness)):
                                self.gameBoard.appealOfWitnesses(self.cardsRecognitionHelper.IsInLineOfSight(self.lastimg))
                    else:
                        print("Action Pawn not Validated")